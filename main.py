import argparse
import hashlib
import os
from pipeline.process_doc_a import process_document_a
from pipeline.process_doc_b import process_document_b
from utils.parallel import run_parallel
from dotenv import load_dotenv
from functools import partial
from pipeline.compare import compare_documents_to_markdown
from multiprocessing import Process

def hash_filenames(path_a: str, path_b: str) -> str:
    base_a = os.path.basename(path_a)
    base_b = os.path.basename(path_b)
    key = f"{base_a}_{base_b}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]  # short hash

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_a", required=True, help="Path to PDF A")
    parser.add_argument("--doc_b", required=True, help="Path to PDF B")
    args = parser.parse_args()

    job_id = hash_filenames(args.doc_a, args.doc_b)

    fn_a = partial(process_document_a, args.doc_a, job_id)
    fn_b = partial(process_document_b, args.doc_b, job_id)

    def run_all(job_id, fn_a, fn_b):
        p1 = Process(target=fn_a)
        p2 = Process(target=fn_b)
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        # Now compare
        compare_documents_to_markdown(job_id)

    run_all(job_id, fn_a, fn_b)

if __name__ == "__main__":
    load_dotenv()
    main()
