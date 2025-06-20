import argparse
import hashlib
import os
from dotenv import load_dotenv
load_dotenv()

from pipeline.chunk_and_store_document import chunk_and_store_document, process_multiple_document_directories
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
    parser.add_argument("--search-mode", choices=["topic_only", "hybrid", "topic_aware"], default="hybrid",
                        help="Search strategy for comparison: topic_only, hybrid, or topic_aware")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process from each document")
    args = parser.parse_args()

    job_id = hash_filenames(args.doc_a, args.doc_b)
    doc_a_name = os.path.basename(args.doc_a)
    doc_b_name = os.path.basename(args.doc_b)

    if args.mode == "index_and_compare":
        # Chunk documents and save to working directories
        print("Chunking document A...")
        doc_a_dir = chunk_and_store_document(args.doc_a, job_id, "document_a", False, max_pages=args.max_pages)
        print(f"Document A chunks saved to: {doc_a_dir}")
        
        print("Chunking document B...")
        doc_b_dir = chunk_and_store_document(args.doc_b, job_id, "document_b", False, max_pages=args.max_pages)
        print(f"Document B chunks saved to: {doc_b_dir}")
        
        # Process chunks and embed them in the database
        print("Processing chunks and generating embeddings...")
        process_multiple_document_directories([doc_a_dir, doc_b_dir])
        print("Embedding and storage completed.")

    # Compare documents using the specified search mode
    print(f"Comparing documents using {args.search_mode} search mode...")
    compare_documents_to_markdown(job_id, doc_a_name, doc_b_name, args.search_mode)
    print(f"[Info] Comparison saved to: comparison_{job_id}_{args.search_mode}.md")

if __name__ == "__main__":
    main()
