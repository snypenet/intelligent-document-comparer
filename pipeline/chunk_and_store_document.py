from collections import deque
import os
from openai import OpenAI
from .extract import extract_pages
from .embed import embed_text
from .db import store_embedding, exists
from tqdm import tqdm
import json

def chunk_with_llm_single_page(page_text: str, page_number: int) -> list[dict]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a document analyst. Chunk the following page into coherent and meaningful sections.
Each chunk should:
- Contain a portion of the text that makes sense on its own
- Be complete, not cutting off sentences or ideas
- Reference the page number {page_number}

Return only JSON in the following format:
[
  {{
    "chunk_text": "...",
    "page_numbers": [page_number]
  }}
]

Here is the page content:

{page_text}
"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {e}\n\nResponse:\n{content}")

def chunk_and_store_document(path: str, file_id: str, index_type: str, use_sliding_window: bool = False, window_size: int = 3):
    pages = extract_pages(path)
    buffer = deque(maxlen=window_size)
    chunk_counter = 0

    # Pre-fill buffer for windowing
    for i, page in enumerate(tqdm(pages, desc=f"[{index_type.upper()}] Chunking Pages", unit="pg", ascii=True)):
        try:
            page_chunks = chunk_with_llm_single_page(page, i)
            for chunk in page_chunks:
                buffer.append(chunk)
                # Only process when buffer is full (centered window)
                if use_sliding_window and len(buffer) == window_size:
                    # Center chunk is at position 1 for window size 3
                    merged_text = "\n".join([c["chunk_text"] for c in buffer])
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
                    merged_text = chunk["chunk_text"]
                    pages_in_window = chunk["page_numbers"]
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
            print(f"[Warning] Failed to chunk page {i}: {e}")

    # After all pages, process any remaining windows in buffer (for sliding window mode)
    if use_sliding_window and len(buffer) > 1:
        # Process trailing windows: e.g. for 3-chunk buffer, process [last-1, last, ...]
        while len(buffer) > 1:
            merged_text = "\n".join([c["chunk_text"] for c in buffer])
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
