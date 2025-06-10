from .extract import extract_pages
from .summarize import summarize_page
from .embed import embed_text
from .db import store_embedding, exists

def process_document_a(path: str, file_id: str):
    pages = extract_pages(path)
    for i, text in enumerate(pages):
        doc_id = f"{file_id}_A_{i}"
        if exists(doc_id):
            print(f"Skipping {doc_id} — already exists.")
            continue

        summary = summarize_page(text)
        embedding = embed_text(summary)
        store_embedding({
            "id": doc_id,
            "file_id": file_id,
            "pages": str(i),
            "text": text,
            "summary": summary,
            "embedding": embedding,
            "index_type": "document_a"
        })
