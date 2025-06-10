from .extract import extract_pages
from .embed import embed_text
from .db import store_embedding, exists

def process_document_b(path: str, file_id: str):
    pages = extract_pages(path)
    for i in range(len(pages)):
        doc_id = f"{file_id}_B_{i}"
        if exists(doc_id):
            print(f"Skipping {doc_id} — already exists.")
            continue

        chunk = "\n".join(pages[max(0, i - 1):min(len(pages), i + 2)])
        embedding = embed_text(chunk)
        store_embedding({
            "id": doc_id,
            "file_id": file_id,
            "pages": ",".join(map(str, [max(0, i - 1), i, min(len(pages) - 1, i + 1)])),
            "text": chunk,
            "embedding": embedding,
            "index_type": "document_b"
        })
