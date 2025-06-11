import argparse
import hashlib
import os
from dotenv import load_dotenv
load_dotenv()

from multiprocessing import Process
from functools import partial
from pipeline.chunk_and_store_document import chunk_and_store_document
from pipeline.compare import compare_documents_to_markdown

def hash_filenames(path_a: str, path_b: str) -> str:
    base_a = os.path.basename(path_a)
    base_b = os.path.basename(path_b)
    key = f"{base_a}_{base_b}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]  # short hash

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_a", required=True, help="Path to PDF A")
    parser.add_argument("--doc_b", required=True, help="Path to PDF B")
    parser.add_argument("--mode", choices=["index_and_compare", "compare_only"], default="index_and_compare",
                        help="Mode to run: either index_and_compare or compare_only")
    args = parser.parse_args()

    job_id = hash_filenames(args.doc_a, args.doc_b)

    if args.mode == "index_and_compare":
        chunk_and_store_document(args.doc_a, job_id, "document_a", False)
        chunk_and_store_document(args.doc_b, job_id, "document_b", True)

    compare_documents_to_markdown(job_id)

if __name__ == "__main__":
    main()
