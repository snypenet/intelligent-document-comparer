from collections import deque
import os
from openai import OpenAI
from .extract import extract_pages
from .embed import embed_text
from .db import store_embedding, exists
from tqdm import tqdm
import traceback
from .utils import call_with_retries

def chunk_with_llm_single_page(page_text: str) -> list[dict]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
You are a document analyst. Given the following page from a document, extract a concise list of the main ideas, topics, or rules discussed on this page. Ignore boilerplate, headers, and footers. Only keep semantically distinct topics. 

Return only new topics not semantically similar to any topics in the provided list, separated by the token "||||". 

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

def chunk_and_store_document(path: str, file_id: str, index_type: str, use_sliding_window: bool = False, window_size: int = 3):
    topics_path = f"topics_{file_id}_{index_type}.txt"
    other_index_type = "document_b" if index_type == "document_a" else "document_a"
    other_topics_path = f"topics_{file_id}_{other_index_type}.txt"
    merged_topics_path = f"topics_{file_id}_merged.txt"

    # If topics file exists, load topics and skip extraction, but still chunk and embed
    topics_exist = os.path.exists(topics_path)
    if topics_exist:
        print(f"[Info] Topics file exists for {index_type}, skipping topic extraction: {topics_path}")
        with open(topics_path, "r", encoding="utf-8") as f:
            all_topics = [line.strip() for line in f if line.strip()]
    else:
        all_topics = []

    pages = extract_pages(path)
    buffer = deque(maxlen=window_size)
    chunk_counter = 0

    for i, page in enumerate(tqdm(pages, desc=f"[{index_type.upper()}] Chunking Pages", unit="pg", ascii=True)):
        try:
            # Topic extraction step (only if topics file does not exist)
            if not topics_exist:
                new_topics = extract_topics_with_llm(page, all_topics)
                for topic in new_topics:
                    if topic not in all_topics:
                        all_topics.append(topic)

            page_chunks = chunk_with_llm_single_page(page)
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
            traceback.print_exc()
            print(f"[Warning] Failed to chunk page {i}: {e}")

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
