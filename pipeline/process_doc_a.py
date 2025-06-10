from .extract import extract_pages
from .summarize import summarize_page
from .embed import embed_text
from .db import store_embedding, exists
from tqdm import tqdm

def process_document_a(path: str, file_id: str):
    pages = extract_pages(path)
    skipped = 0
    for i, text in enumerate(tqdm(pages, desc=f"[A] Indexing {file_id}", unit="pg", ascii=True, position=0)):
        doc_id = f"{file_id}_A_{i}"
        if exists(doc_id):
            skipped += 1
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

    if skipped:
        print(f"[A] Skipped {skipped} previously indexed pages.")

