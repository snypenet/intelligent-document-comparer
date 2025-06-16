import os
import json
import chromadb
from openai import OpenAI
from .embed import embed_text
import numpy as np
import time
from .utils import call_with_retries


def compare_documents_to_markdown(job_id: str, doc_a_name: str, doc_b_name: str):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("document_embeddings")
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Read topics from merged file if it exists, otherwise fallback to document_a
    merged_topics_path = f"topics_{job_id}_merged.txt"
    topics_path = f"topics_{job_id}_document_a.txt"
    if os.path.exists(merged_topics_path):
        with open(merged_topics_path, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        with open(topics_path, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]

    def get_best_chunks(topic, collection, job_id, index_type, n_results=3):
        topic_emb = embed_text(topic)
        result = collection.query(
            query_embeddings=topic_emb,
            where={"$and": [{"index_type": index_type}, {"file_id": job_id}]},
            n_results=n_results
        )
        if not result["documents"] or not result["documents"][0]:
            print(f"[Warning] No match found for topic: {topic} in {index_type}")
            return [], [], []
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        ids = result["ids"][0]
        return docs, metas, ids

    output_path = f"comparison_{job_id}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Topic-Driven Differences\n\n")
        f.write(f"Comparing:\n- Document A: {doc_a_name}\n- Document B: {doc_b_name}\n\n")

    for topic in topics:
        # Find best matching chunks for this topic in both docs using ChromaDB query
        docs_a, metas_a, ids_a = get_best_chunks(topic, collection, job_id, "document_a")
        docs_b, metas_b, ids_b = get_best_chunks(topic, collection, job_id, "document_b")
        missing_a = not docs_a
        missing_b = not docs_b
        if missing_a and missing_b:
            print(f"[Warning] Skipping topic due to missing chunks in both documents: {topic}")
            continue

        def format_chunks(docs, metas):
            return "\n---\n".join([
                f"(pages {meta.get('pages')}):\n{doc}" for doc, meta in zip(docs, metas)
            ])

        doc_a_str = format_chunks(docs_a, metas_a) if not missing_a else f"[No relevant content found in {doc_a_name} for this topic.]"
        doc_b_str = format_chunks(docs_b, metas_b) if not missing_b else f"[No relevant content found in {doc_b_name} for this topic.]"

        prompt = f"""You are a document comparison analyst.\n\nFor the topic below, compare {doc_a_name} and {doc_b_name}.\n\n- Summarize only the substantive differences in rules, requirements, or content.\n- Be concise and direct; avoid unnecessary elaboration.\n- Always include any relevant numbers, figures, or timelines from both documents in the summary.\n- Reference the page numbers for each difference.\n- Ignore minor wording or formatting changes.\n\nTopic: \"{topic}\"\n\n{doc_a_name} (top 3 relevant chunks):\n{doc_a_str}\n\n{doc_b_name} (top 3 relevant chunks):\n{doc_b_str}\n"""

        # Use retry wrapper for OpenAI call
        response = call_with_retries(lambda: llm.chat.completions.create(
            model=os.getenv("MODEL_TYPE"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        ))

        summary = response.choices[0].message.content.strip()

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"## {topic}\n\n{summary}\n\n---\n")

    print(f"Topic-driven markdown diff report saved to: {output_path}")

def call_with_retries(func, max_retries=3, initial_delay=2, *args, **kwargs):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[Retry] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2  # Exponential backoff
