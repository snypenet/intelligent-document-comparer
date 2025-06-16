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
        temperature=0.3
    ))
    content = response.choices[0].message.content.strip()
    # Split and clean topics
    new_topics = [t.strip() for t in content.split("||||") if t.strip()]
    return new_topics

def further_refine_topics_with_llm(topics: list[str]) -> set[str]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a document analyst refining a list of topics. Take the "List of Topics" and refine them by merging any semantically similar topics into a single broader topic that still captures the core ideas. The final list of topics will be used to power a document search and comparison between two similar documents.

Return only the refined list of topics, separated by the delimiter "||||".

List of Topics:
    {chr(10).join(topics)}
    """
    response = call_with_retries(lambda: client.chat.completions.create(
        model=os.getenv("MODEL_TYPE"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    ))
    content = response.choices[0].message.content.strip()
    refined_topics = {t.strip() for t in content.split("||||") if t.strip()}
    return refined_topics

def merge_topics_with_llm(topics_a: list[str], topics_b: list[str]) -> list[str]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a document analyst. Given two lists of topics extracted from two related documents, produce a single, distinct, concise list of all unique topics that exist across both documents. Remove duplicates, merge similar topics, and ensure the list is comprehensive but not redundant. Only keep semanticallty different topics.Return the topics separated by the token "||||".

Example:
Topic 1||||Topic 2||||Topic 3||||Topic 4

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

def chunk_and_store_document(path: str, file_id: str, index_type: str, use_sliding_window: bool = False, window_size: int = 3, max_pages: int = None):
    # Create run directory with GUID
    run_id = str(uuid.uuid4())
    run_dir = Path(f"runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")

    # Create document-specific directory
    doc_dir = run_dir / f"{file_id}_{index_type}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created document directory: {doc_dir}")

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
                    if not exists(doc_id):
                        embedding = embed_text(merged_text)
                        store_embedding({
                            "id": doc_id,
                            "file_id": file_id,
                            "pages": ",".join(map(str, pages_in_window)),
                            "text": merged_text,
                            "embedding": embedding,
                            "index_type": index_type
                        })
                    chunk_counter += 1
                elif not use_sliding_window:
                    # No windowing: process and store immediately
                    merged_text = chunk
                    if not merged_text.strip():
                        continue  # Skip empty chunk
                    pages_in_window = [i+1]
                    doc_id = f"{file_id}_{index_type}_{chunk_counter}"
                    if not exists(doc_id):
                        embedding = embed_text(merged_text)
                        store_embedding({
                            "id": doc_id,
                            "file_id": file_id,
                            "pages": ",".join(map(str, pages_in_window)),
                            "text": merged_text,
                            "embedding": embedding,
                            "index_type": index_type
                        })
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
            if not exists(doc_id):
                embedding = embed_text(merged_text)
                store_embedding({
                    "id": doc_id,
                    "file_id": file_id,
                    "pages": ",".join(map(str, pages_in_window)),
                    "text": merged_text,
                    "embedding": embedding,
                    "index_type": index_type
                })
            buffer.popleft()  # Slide window forward
            chunk_counter += 1

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
