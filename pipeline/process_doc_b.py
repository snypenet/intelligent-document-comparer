from .extract import extract_pages
from .embed import embed_text
from .db import store_embedding, exists
from tqdm import tqdm

def process_document_b(path: str, file_id: str):
    pages = extract_pages(path)
    total = len(pages)
    skipped = 0
    for i in tqdm(range(total), desc=f"[B] Indexing {file_id}", unit="pg", ascii=True, position=1):
        doc_id = f"{file_id}_B_{i}"
        if exists(doc_id):
            skipped += 1
            continue

        chunk = "\n".join(pages[max(0, i - 1):min(total, i + 2)])
        embedding = embed_text(chunk)
        store_embedding({
            "id": doc_id,
            "file_id": file_id,
            "pages": ",".join(map(str, [max(0, i - 1), i, min(total - 1, i + 1)])),
            "text": chunk,
            "embedding": embedding,
            "index_type": "document_b"
        })

    if skipped:
        print(f"[B] Skipped {skipped} previously indexed chunks.")
