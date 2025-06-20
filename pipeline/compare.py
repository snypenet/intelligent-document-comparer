import os
import json
from openai import OpenAI
from .db import query_by_topics, hybrid_search, topic_aware_semantic_search
import time
from .utils import call_with_retries
from pathlib import Path


def compare_documents_to_markdown(job_id: str, doc_a_name: str, doc_b_name: str, 
                                 search_mode: str = "topic_only"):
    """
    Compare documents using different search strategies.
    
    Args:
        job_id: Unique identifier for the document pair
        doc_a_name: Name of document A
        doc_b_name: Name of document B
        search_mode: Search strategy to use:
            - "topic_only": Use only topic-based filtering
            - "hybrid": Combine topic filtering with embedding similarity
            - "topic_aware": Semantic search with topic relevance bonus
    """
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create run directory for logs
    run_dir = Path(f"runs/{job_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "comparison_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Read topics from merged file if it exists, otherwise fallback to document_a
    merged_topics_path = f"topics_{job_id}_merged.txt"
    topics_path = f"topics_{job_id}_document_a.txt"
    if os.path.exists(merged_topics_path):
        with open(merged_topics_path, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        with open(topics_path, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]

    def get_chunks_by_search_mode(topic, job_id, index_type, n_results=5):
        """
        Get chunks using the specified search mode.
        """
        where_clause = {"$and": [{"index_type": index_type}, {"file_id": job_id}]}
        
        try:
            if search_mode == "topic_only":
                # Pure topic-based filtering
                result = query_by_topics(
                    topics=[topic],
                    n_results=n_results,
                    where=where_clause
                )
            elif search_mode == "hybrid":
                # Hybrid search: topic filtering + embedding similarity
                result = hybrid_search(
                    query_text=topic,
                    topics=[topic],
                    n_results=n_results,
                    where=where_clause,
                    topic_weight=0.4,
                    embedding_weight=0.6
                )
            elif search_mode == "topic_aware":
                # Semantic search with topic relevance bonus
                result = topic_aware_semantic_search(
                    query_text=topic,
                    topics=[topic],
                    n_results=n_results,
                    where=where_clause,
                    topic_threshold=0.5
                )
            else:
                raise ValueError(f"Unknown search mode: {search_mode}")
            
            if not result["documents"] or not result["documents"][0]:
                print(f"[Warning] No chunks found for topic '{topic}' in {index_type} using {search_mode} mode")
                return [], [], []
            
            docs = result["documents"][0]
            metas = result["metadatas"][0]
            ids = result["ids"][0]
            return docs, metas, ids
            
        except Exception as e:
            print(f"[Error] Failed to query chunks for topic '{topic}' in {index_type} using {search_mode} mode: {e}")
            return [], [], []

    output_path = f"comparison_{job_id}_{search_mode}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Topic-Driven Differences ({search_mode.replace('_', ' ').title()} Mode)\n\n")
        f.write(f"Comparing:\n- Document A: {doc_a_name}\n- Document B: {doc_b_name}\n\n")
        f.write(f"Search Mode: {search_mode}\n\n")
        
        # Add alphabetized list of topics with links
        f.write("## Topics Covered\n\n")
        for topic in sorted(topics):
            # Create anchor link by converting topic to lowercase and replacing spaces with hyphens
            anchor = topic.lower().replace(" ", "-")
            f.write(f"- [{topic}](#{anchor})\n")
        f.write("\n---\n\n")

    # Process topics in alphabetical order
    for topic in sorted(topics):
        # Create topic-specific log file
        topic_log_path = logs_dir / f"{topic.lower().replace(' ', '_')}_{search_mode}_log.txt"
        with open(topic_log_path, "w", encoding="utf-8") as log:
            log.write(f"Topic: {topic}\n")
            log.write(f"Search Mode: {search_mode}\n")
            log.write("=" * 80 + "\n\n")

            # Find chunks using the specified search mode
            docs_a, metas_a, ids_a = get_chunks_by_search_mode(topic, job_id, "document_a", 5)
            docs_b, metas_b, ids_b = get_chunks_by_search_mode(topic, job_id, "document_b", 5)
            
            # Log search results
            log.write(f"Search Results for {doc_a_name}:\n")
            log.write("-" * 40 + "\n")
            for doc, meta in zip(docs_a, metas_a):
                log.write(f"Pages: {meta.get('pages')}\n")
                log.write(f"Assigned Topics: {meta.get('assigned_topics', 'N/A')}\n")
                if search_mode != "topic_only":
                    log.write(f"Score: {meta.get('score', 'N/A')}\n")
                log.write(f"Content: {doc}\n\n")
            
            log.write(f"Search Results for {doc_b_name}:\n")
            log.write("-" * 40 + "\n")
            for doc, meta in zip(docs_b, metas_b):
                log.write(f"Pages: {meta.get('pages')}\n")
                log.write(f"Assigned Topics: {meta.get('assigned_topics', 'N/A')}\n")
                if search_mode != "topic_only":
                    log.write(f"Score: {meta.get('score', 'N/A')}\n")
                log.write(f"Content: {doc}\n\n")

            missing_a = not docs_a
            missing_b = not docs_b
            if missing_a and missing_b:
                print(f"[Warning] Skipping topic due to missing chunks in both documents: {topic}")
                log.write("Skipped: No chunks found with this topic in either document\n")
                continue

            def format_chunks(docs, metas):
                return "\n---\n".join([
                    f"(pages {meta.get('pages')}):\n{doc}" for doc, meta in zip(docs, metas)
                ])

            doc_a_str = format_chunks(docs_a, metas_a) if not missing_a else f"[No chunks found with topic '{topic}' in {doc_a_name}.]"
            doc_b_str = format_chunks(docs_b, metas_b) if not missing_b else f"[No chunks found with topic '{topic}' in {doc_b_name}.]"

            prompt = f"""
You are a document comparison analyst.

Your job is to identify real differences between two documents, not surface-level noise.

Compare the documents below based on the topic: "{topic}"

Please follow these strict rules:
- Only summarize substantive differences in facts, rules, requirements, or claims.
- If both documents say the same thing, even in slightly different words, do not list it as a difference.
- Ignore repetition, reordering, or phrasing variations.
- Do not infer differences based on chunk boundaries.
- If there are no real differences, clearly respond: "No substantive differences found."

When differences do exist:
- Be concise.
- Include relevant numbers, dates, or figures.
- Always cite the page numbers where the differences appear.

Reference:
- {doc_a_name} (chunks found using {search_mode} search for topic '{topic}'):
{doc_a_str}

- {doc_b_name} (chunks found using {search_mode} search for topic '{topic}'):
{doc_b_str}
            """
            
            # Log the prompt
            log.write("LLM Prompt:\n")
            log.write("-" * 40 + "\n")
            log.write(prompt + "\n\n")

            # Use retry wrapper for OpenAI call
            response = call_with_retries(lambda: llm.chat.completions.create(
                model=os.getenv("SUMMARY_MODEL_TYPE"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            ))

            summary = response.choices[0].message.content.strip()
            
            # Log the LLM response
            log.write("LLM Response:\n")
            log.write("-" * 40 + "\n")
            log.write(summary + "\n")

            with open(output_path, "a", encoding="utf-8") as f:
                # Create anchor for the section
                anchor = topic.lower().replace(" ", "-")
                f.write(f"## <a id='{anchor}'></a>{topic}\n\n{summary}\n\n---\n")

    print(f"Topic-driven markdown diff report saved to: {output_path}")
    print(f"Detailed logs saved to: {logs_dir}")

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
