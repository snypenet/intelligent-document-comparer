import argparse
import hashlib
import os
from dotenv import load_dotenv
load_dotenv()

from pipeline.chunk_and_store_document import chunk_and_store_document
from pipeline.compare import compare_documents_to_markdown

def hash_filenames(path_a: str, path_b: str) -> str:
    base_a = os.path.basename(path_a)
    base_b = os.path.basename(path_b)
    key = f"{base_a}_{base_b}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]  # short hash

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-a", required=True, help="Path to PDF A")
    parser.add_argument("--doc-b", required=True, help="Path to PDF B")
    parser.add_argument("--mode", choices=["index_and_compare", "compare_only"], default="index_and_compare",
                        help="Mode to run: either index_and_compare or compare_only")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process from each document")
    args = parser.parse_args()

    job_id = hash_filenames(args.doc_a, args.doc_b)
    doc_a_name = os.path.basename(args.doc_a)
    doc_b_name = os.path.basename(args.doc_b)

    if args.mode == "index_and_compare":
        chunk_and_store_document(args.doc_a, job_id, "document_a", False, max_pages=args.max_pages)
        chunk_and_store_document(args.doc_b, job_id, "document_b", True, max_pages=args.max_pages)

    compare_documents_to_markdown(job_id, doc_a_name, doc_b_name)
    print(f"[Info] Comparison saved to: comparison_{job_id}.md")

if __name__ == "__main__":
    main()
