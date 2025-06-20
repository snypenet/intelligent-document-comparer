from collections import deque
import os
from openai import OpenAI
from .extract import extract_pages
from .embed import embed_text
from .db import store_embedding, exists
from tqdm import tqdm
import traceback
from .utils import call_with_retries
import uuid
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def chunk_with_llm_single_page(page_text: str, topics: set[str] = None) -> list[dict]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # If topics are provided, use them to filter content
    if topics:
        prompt = f"""
You are a document analyst. Your task is to identify and extract chunks of text that are relevant to the specified topics.
Each chunk should:
- Only contain content that is directly related to one or more of the specified topics
- Be complete and self-contained
- Ignore any content that is not relevant to the specified topics
- Maintain the original text structure and wording

Only extract chunks that are relevant to these topics:
{chr(10).join(f"- {topic}" for topic in topics)}

Return each relevant chunk separated by the token "||||"
Example of returned chunks:
chunk 1||||chunk2||||chunk 3||||chunk 4

Here is the page content:

{page_text}
"""
    else:
        prompt = f"""
You are a document analyst. Chunk the following page into coherent and meaningful sections.
Each chunk should:
- Contain a portion of the text that makes sense on its own
- Be complete, not cutting off sentences or ideas

Return each chunk separated by the token "||||"
Example of returned chunks:
chunk 1||||chunk2||||chunk 3||||chunk 4

Here is the page content:

{page_text}
"""
    response = call_with_retries(lambda: client.chat.completions.create(
        model=os.getenv("MODEL_TYPE"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    ))
    content = response.choices[0].message.content.strip().replace("'", "'").replace('\u2018', "'").replace('"', '"')
    return content.split("||||")

def extract_topics_with_llm(page_text: str, current_topics: list[str]) -> list[str]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a document analyst extracting main thematic topics from the "Page content". Focus on generating a small set of broad, semantically distinct topics that group related subtopics and policy provisions together. Avoid listing many narrowly defined or overlapping topics. Each topic should cover a comprehensive theme to support efficient retrieval and summarization across a large document. Ignore boilerplate, headers, and footers.
Return only new topics not semantically similar to any topics in the "Current list of topics", separated by the delimiter "||||". 

Example:
Topic 1||||Topic 2||||Topic 3||||Topic 4

Current list of topics:
{chr(10).join(current_topics) if current_topics else '(none)'}

Page content:
{page_text}
"""
    response = call_with_retries(lambda: client.chat.completions.create(
        model=os.getenv("MODEL_TYPE"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    ))
    content = response.choices[0].message.content.strip()
    # Split and clean topics
    new_topics = [t.strip() for t in content.split("||||") if t.strip()]
    return new_topics

def further_refine_topics_with_llm(topics: list[str]) -> set[str]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a senior document analyst preparing a refined topic index to support semantic document comparison via Retrieval-Augmented Generation (RAG). Your task is to take the "All Topics" list and consolidate semantically related entries into a concise set of broader topic categories.

Each final topic must:

    Represent a high-level umbrella that generalizes multiple specific themes.

    Be mutually exclusive from the others.

    Be broad enough to group related topics for retrieval, but not so broad that it loses meaning.

    Minimize redundancy and merge any topics that share common regulatory, payment, or care-setting considerations.

Return only the final list of consolidated topic labels, separated by the delimiter ||||.
    {chr(10).join(topics)}
    """
    response = call_with_retries(lambda: client.chat.completions.create(
        model=os.getenv("MODEL_TYPE"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    ))
    content = response.choices[0].message.content.strip()
    refined_topics = {t.strip() for t in content.split("||||") if t.strip()}
    return refined_topics

def merge_topics_with_llm(topics_a: list[str], topics_b: list[str]) -> list[str]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a domain-aware document analyst. Given two lists of topics extracted from related documents, produce a single, comprehensive list of semantically unique topics.

Merge overlapping or closely related topics into a single, representative phrasing. Eliminate redundancy without omitting distinct ideas.

Focus on semantic distinctiveness, not just surface differences.

Return the final list of consolidated topics, separated by the token "||||" - one topic per entry.

Only output the final list.

Example:
Topic A||||Topic B||||Topic C||||Topic D

List A:
{chr(10).join(topics_a)}

List B:
{chr(10).join(topics_b)}
"""
    response = call_with_retries(lambda: client.chat.completions.create(
        model=os.getenv("MODEL_TYPE"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    ))
    content = response.choices[0].message.content.strip()
    merged_topics = [t.strip() for t in content.split("||||") if t.strip()]
    return merged_topics

def chunk_and_store_document(path: str, file_id: str, index_type: str, use_sliding_window: bool = False, window_size: int = 3, max_pages: int = None) -> Path:
    # Create run directory with GUID
    run_id = str(uuid.uuid4())
    run_dir = Path(f"runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")

    # Create document-specific directory
    doc_dir = run_dir / f"{file_id}_{index_type}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created document directory: {doc_dir}")

    # Create chunks directory for storing chunk metadata
    chunks_dir = doc_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created chunks directory: {chunks_dir}")

    topics_path = f"topics_{file_id}_{index_type}.txt"
    other_index_type = "document_b" if index_type == "document_a" else "document_a"
    other_topics_path = f"topics_{file_id}_{other_index_type}.txt"
    merged_topics_path = f"topics_{file_id}_merged.txt"

    # If topics file exists, load topics and skip extraction, but still chunk and embed
    topics_exist = os.path.exists(topics_path)
    if topics_exist:
        logger.info(f"Topics file exists for {index_type}, skipping topic extraction: {topics_path}")
        with open(topics_path, "r", encoding="utf-8") as f:
            all_topics = set({line.strip() for line in f if line.strip()})
    else:
        all_topics = set()

    pages = extract_pages(path)
    if max_pages is not None:
        pages = pages[:max_pages]
        logger.info(f"Limiting processing to {max_pages} pages")
    
    buffer = deque(maxlen=window_size)
    chunk_counter = 0
    chunk_metadata = []

    for i, page in enumerate(tqdm(pages, desc=f"[{index_type.upper()}] Chunking Pages", unit="pg", ascii=True)):
        try:
            # Create page-specific directory
            page_dir = doc_dir / f"page_{i+1}"
            page_dir.mkdir(parents=True,exist_ok=True)
            logger.info(f"Processing page {i+1} - Output directory: {page_dir}")

            # Topic extraction step (only if topics file does not exist)
            if not topics_exist:
                new_topics = extract_topics_with_llm(page, all_topics)
                # Save initial topics
                with open(page_dir / "topics.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(new_topics))
                logger.info(f"Saved initial topics for page {i+1}")

                for topic in new_topics:
                    if topic not in all_topics:
                        all_topics.add(topic)

                # have the LLM check it's own work
                all_topics = further_refine_topics_with_llm(all_topics)
                # Save refined topics
                with open(page_dir / "refined_all_topics.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(all_topics))
                logger.info(f"Saved refined all topics for page {i+1}")

            page_chunks = chunk_with_llm_single_page(page, all_topics)
            # Save chunks
            with open(page_dir / "chunks.txt", "w", encoding="utf-8") as f:
                f.write("\n\n---CHUNK---\n\n".join(page_chunks))
            logger.info(f"Saved {len(page_chunks)} chunks for page {i+1}")

            for chunk in page_chunks:
                buffer.append({
                    "chunk_text": chunk,
                    "page_numbers": [i+1]
                })
                # Only process when buffer is full (centered window)
                if use_sliding_window and len(buffer) == window_size:
                    # Center chunk is at position 1 for window size 3
                    merged_text = "\n".join([c["chunk_text"] for c in buffer])
                    if not merged_text.strip():
                        continue  # Skip empty merged chunk
                    pages_in_window = sorted({pn for c in buffer for pn in c["page_numbers"]})
                    doc_id = f"{file_id}_{index_type}_{chunk_counter}"
                    
                    # Save chunk metadata instead of embedding immediately
                    chunk_data = {
                        "id": doc_id,
                        "file_id": file_id,
                        "pages": ",".join(map(str, pages_in_window)),
                        "text": merged_text,
                        "index_type": index_type,
                        "chunk_number": chunk_counter,
                        "use_sliding_window": use_sliding_window,
                        "window_size": window_size,
                        "topics": list(all_topics) if all_topics else []
                    }
                    
                    # Save chunk text to file
                    chunk_file = chunks_dir / f"chunk_{chunk_counter:06d}.txt"
                    with open(chunk_file, "w", encoding="utf-8") as f:
                        f.write(merged_text)
                    
                    # Save metadata
                    chunk_data["text_file"] = str(chunk_file)
                    chunk_metadata.append(chunk_data)
                    
                    chunk_counter += 1
                elif not use_sliding_window:
                    # No windowing: process and store immediately
                    merged_text = chunk
                    if not merged_text.strip():
                        continue  # Skip empty chunk
                    pages_in_window = [i+1]
                    doc_id = f"{file_id}_{index_type}_{chunk_counter}"
                    
                    # Save chunk metadata instead of embedding immediately
                    chunk_data = {
                        "id": doc_id,
                        "file_id": file_id,
                        "pages": ",".join(map(str, pages_in_window)),
                        "text": merged_text,
                        "index_type": index_type,
                        "chunk_number": chunk_counter,
                        "use_sliding_window": use_sliding_window,
                        "window_size": window_size,
                        "topics": list(all_topics) if all_topics else []
                    }
                    
                    # Save chunk text to file
                    chunk_file = chunks_dir / f"chunk_{chunk_counter:06d}.txt"
                    with open(chunk_file, "w", encoding="utf-8") as f:
                        f.write(merged_text)
                    
                    # Save metadata
                    chunk_data["text_file"] = str(chunk_file)
                    chunk_metadata.append(chunk_data)
                    
                    chunk_counter += 1
        except Exception as e:
            logger.error(f"Failed to process page {i+1}: {str(e)}")
            traceback.print_exc()

    # After all pages, process any remaining windows in buffer (for sliding window mode)
    if use_sliding_window and len(buffer) > 1:
        # Process trailing windows: e.g. for 3-chunk buffer, process [last-1, last, ...]
        while len(buffer) > 1:
            merged_text = "\n".join([c["chunk_text"] for c in buffer])
            if not merged_text.strip():
                buffer.popleft()
                continue  # Skip empty merged chunk
            pages_in_window = sorted({pn for c in buffer for pn in c["page_numbers"]})
            doc_id = f"{file_id}_{index_type}_{chunk_counter}"
            
            # Save chunk metadata instead of embedding immediately
            chunk_data = {
                "id": doc_id,
                "file_id": file_id,
                "pages": ",".join(map(str, pages_in_window)),
                "text": merged_text,
                "index_type": index_type,
                "chunk_number": chunk_counter,
                "use_sliding_window": use_sliding_window,
                "window_size": window_size,
                "topics": list(all_topics) if all_topics else []
            }
            
            # Save chunk text to file
            chunk_file = chunks_dir / f"chunk_{chunk_counter:06d}.txt"
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(merged_text)
            
            # Save metadata
            chunk_data["text_file"] = str(chunk_file)
            chunk_metadata.append(chunk_data)
            
            buffer.popleft()  # Slide window forward
            chunk_counter += 1

    # Save chunk metadata to JSON file
    metadata_file = doc_dir / "chunk_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved chunk metadata to {metadata_file}")

    # Save the final topic list to a file for later use (only if it didn't exist)
    if not topics_exist:
        with open(topics_path, "w", encoding="utf-8") as f:
            for topic in all_topics:
                f.write(topic + "\n")
        logger.info(f"Saved final topics to {topics_path}")

    # If both topic files exist and merged file does not, merge and deduplicate them using the LLM
    if os.path.exists(topics_path) and os.path.exists(other_topics_path) and not os.path.exists(merged_topics_path):
        with open(topics_path, "r", encoding="utf-8") as f:
            topics_this = [line.strip() for line in f if line.strip()]
        with open(other_topics_path, "r", encoding="utf-8") as f:
            topics_other = [line.strip() for line in f if line.strip()]
        merged_topics = merge_topics_with_llm(topics_this, topics_other)
        with open(merged_topics_path, "w", encoding="utf-8") as f:
            for topic in merged_topics:
                f.write(topic + "\n")
        logger.info(f"Saved merged topics to {merged_topics_path}")

    return doc_dir

def assign_topics_to_chunk_with_llm(chunk_text: str, available_topics: list[str]) -> list[str]:
    """
    Use LLM to assign 0 or more relevant topics to a chunk from the available topics list.
    
    Args:
        chunk_text: The text content of the chunk
        available_topics: List of available topics to choose from
        
    Returns:
        List of assigned topic names (can be empty)
    """
    if not available_topics:
        return []
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a careful and domain-aware document analyst. Your task is to assign relevant topics to a text chunk from a predefined list of topics.

Available topics:
{chr(10).join(f"- {topic}" for topic in available_topics)}
Text chunk:
{chunk_text}

Instructions:

    Assign 0 or more topics from the list that are clearly relevant to the content of the chunk.

    Relevance includes explicit mentions and strongly implied themes (e.g., impact, justice, strategy, innovation).

    If a topic's idea is discussed indirectly but central to the chunk, you should include it.

    Do not assign a topic unless it's well-supported by the content (explicitly or semantically).

    It's better to include a semantically relevant topic than to omit it just because the exact phrase wasn't used.

    If no topics are relevant, return an empty string.

Response format:

    Return only topic names from the list, separated by ||||

    If none are relevant, return a blank line (no quotes)

Example responses:
Topic A||||Topic B||||Topic C
Topic A
(empty)
"""
    
    response = call_with_retries(lambda: client.chat.completions.create(
        model=os.getenv("SUMMARY_MODEL_TYPE"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    ))
    
    content = response.choices[0].message.content.strip()
    if not content:
        return []
    
    # Split and clean assigned topics
    assigned_topics = [t.strip() for t in content.split("||||") if t.strip()]
    
    # Validate that all assigned topics exist in the available topics list
    valid_topics = [topic for topic in assigned_topics if topic in available_topics]
    
    return valid_topics

def review_assigned_topics_with_llm(chunk_text: str, initial_topics: list[str], available_topics: list[str]) -> list[str]:
    """
    Use LLM to review the initial topic assignment for a chunk and either confirm or revise the list from the available topics.
    """
    if not available_topics or not initial_topics:
        return initial_topics or []
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a careful and domain-aware document analyst. You have already assigned the following topics to a text chunk from a predefined list of topics:

Initial assigned topics:
{chr(10).join(f"- {topic}" for topic in initial_topics)}

Available topics:
{chr(10).join(f"- {topic}" for topic in available_topics)}

Text chunk:
{chunk_text}

Instructions:
    Review the initial assigned topics. If they are all appropriate and no others are needed, return the same list.
    If you believe some topics should be removed or others from the available list should be added, return a revised list.
    Only use topics from the available list. Do not invent new topics.
    If no topics are relevant, return a blank line (no quotes).

Response format:
    Return only topic names from the list, separated by ||||
    If none are relevant, return a blank line (no quotes)

Example responses:
Topic A||||Topic B
Topic A
(empty)
"""
    response = call_with_retries(lambda: client.chat.completions.create(
        model=os.getenv("SUMMARY_MODEL_TYPE"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    ))
    content = response.choices[0].message.content.strip()
    if not content:
        return []
    reviewed_topics = [t.strip() for t in content.split("||||") if t.strip()]
    valid_topics = [topic for topic in reviewed_topics if topic in available_topics]
    return valid_topics

def process_multiple_document_directories(doc_dirs: list[Path]) -> None:
    """
    Process chunks from multiple document directories using a shared merged topics file.
    
    Args:
        doc_dirs: List of paths to document directories containing chunk metadata
    """
    logger.info(f"Processing chunks from {len(doc_dirs)} document directories")
    
    # Try to find a merged topics file from any of the directories
    available_topics = []
    merged_topics_found = False
    
    for doc_dir in doc_dirs:
        if not doc_dir.exists():
            logger.warning(f"Document directory does not exist: {doc_dir}")
            continue
        
        # Load chunk metadata to get file_id
        metadata_file = doc_dir / "chunk_metadata.json"
        if not metadata_file.exists():
            logger.warning(f"Chunk metadata file not found: {metadata_file}")
            continue
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            chunk_metadata = json.load(f)
        
        if not chunk_metadata:
            logger.warning(f"No chunk metadata found in {doc_dir}")
            continue
        
        file_id = chunk_metadata[0]["file_id"]
        
        # Try to read merged topics file
        merged_topics_path = f"topics_{file_id}_merged.txt"
        if os.path.exists(merged_topics_path):
            logger.info(f"Found merged topics file: {merged_topics_path}")
            with open(merged_topics_path, "r", encoding="utf-8") as f:
                available_topics = [line.strip() for line in f if line.strip()]
            merged_topics_found = True
            break
    
    # If no merged topics file found, try to use document_a topics as fallback
    if not merged_topics_found:
        for doc_dir in doc_dirs:
            if not doc_dir.exists():
                continue
            
            metadata_file = doc_dir / "chunk_metadata.json"
            if not metadata_file.exists():
                continue
            
            with open(metadata_file, "r", encoding="utf-8") as f:
                chunk_metadata = json.load(f)
            
            if not chunk_metadata:
                continue
            
            file_id = chunk_metadata[0]["file_id"]
            topics_path = f"topics_{file_id}_document_a.txt"
            
            if os.path.exists(topics_path):
                logger.info(f"Using fallback topics file: {topics_path}")
                with open(topics_path, "r", encoding="utf-8") as f:
                    available_topics = [line.strip() for line in f if line.strip()]
                break
    
    if not available_topics:
        logger.warning("No topics file found. Using empty topic list.")
    
    logger.info(f"Using {len(available_topics)} topics for categorization across all documents")
    
    # Process each directory using the shared topics
    for doc_dir in doc_dirs:
        if not doc_dir.exists():
            logger.warning(f"Document directory does not exist: {doc_dir}")
            continue
        
        logger.info(f"Processing document directory: {doc_dir}")
        process_chunks_from_directory_with_topics(doc_dir, available_topics)
    
    logger.info("Completed processing all document directories")

def process_chunks_from_directory_with_topics(doc_dir: Path, available_topics: list[str]) -> None:
    """
    Process chunks from a document directory using provided topics.
    
    Args:
        doc_dir: Path to the document directory containing chunk metadata
        available_topics: List of topics to use for categorization
    """
    metadata_file = doc_dir / "chunk_metadata.json"
    if not metadata_file.exists():
        logger.error(f"Chunk metadata file not found: {metadata_file}")
        return
    
    # Load chunk metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)
    
    logger.info(f"Processing {len(chunk_metadata)} chunks from {doc_dir}")
    
    for chunk_data in tqdm(chunk_metadata, desc="Embedding chunks", unit="chunk"):
        try:
            # Check if chunk already exists in database
            if exists(chunk_data["id"]):
                logger.debug(f"Chunk {chunk_data['id']} already exists, skipping")
                continue
            
            # Read chunk text from file
            text_file = Path(chunk_data["text_file"])
            if not text_file.exists():
                logger.warning(f"Chunk text file not found: {text_file}")
                continue
            
            with open(text_file, "r", encoding="utf-8") as f:
                chunk_text = f.read()
            
            # First LLM call: assign topics
            initial_assigned_topics = assign_topics_to_chunk_with_llm(chunk_text, available_topics)
            # Second LLM call: review/confirm topics
            assigned_topics = review_assigned_topics_with_llm(chunk_text, initial_assigned_topics, available_topics)
            
            # Generate embedding
            embedding = embed_text(chunk_text)
            
            # Store in database with assigned topics
            store_embedding({
                "id": chunk_data["id"],
                "file_id": chunk_data["file_id"],
                "pages": chunk_data["pages"],
                "text": chunk_text,
                "embedding": embedding,
                "index_type": chunk_data["index_type"],
                "assigned_topics": "||||".join(assigned_topics) if assigned_topics else ""
            })
            
            logger.debug(f"Successfully processed chunk {chunk_data['id']} with {len(assigned_topics)} assigned topics (after review)")
            
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_data.get('id', 'unknown')}: {str(e)}")
            traceback.print_exc()
    
    logger.info(f"Completed processing chunks from {doc_dir}")

def process_chunks_from_directory(doc_dir: Path) -> None:
    """
    Process chunks from a document directory and embed/store them in the database.
    This function reads topics from the merged topics file for consistency.
    
    Args:
        doc_dir: Path to the document directory containing chunk metadata
    """
    metadata_file = doc_dir / "chunk_metadata.json"
    if not metadata_file.exists():
        logger.error(f"Chunk metadata file not found: {metadata_file}")
        return
    
    # Load chunk metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)
    
    # Get file_id from the first chunk to construct the merged topics file path
    if not chunk_metadata:
        logger.error("No chunk metadata found")
        return
    
    file_id = chunk_metadata[0]["file_id"]
    
    # Read topics from the merged topics file
    merged_topics_path = f"topics_{file_id}_merged.txt"
    topics_path = f"topics_{file_id}_document_a.txt"  # fallback
    
    available_topics = []
    if os.path.exists(merged_topics_path):
        logger.info(f"Reading topics from merged file: {merged_topics_path}")
        with open(merged_topics_path, "r", encoding="utf-8") as f:
            available_topics = [line.strip() for line in f if line.strip()]
    elif os.path.exists(topics_path):
        logger.info(f"Reading topics from fallback file: {topics_path}")
        with open(topics_path, "r", encoding="utf-8") as f:
            available_topics = [line.strip() for line in f if line.strip()]
    else:
        logger.warning(f"No topics file found. Using empty topic list.")
        available_topics = []
    
    logger.info(f"Loaded {len(available_topics)} topics for categorization")
    
    # Use the shared function with the loaded topics
    process_chunks_from_directory_with_topics(doc_dir, available_topics)
