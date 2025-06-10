import chromadb
from openai import OpenAI
from .embed import embed_text
import os

from openai import OpenAI
import os

def compare_documents_to_markdown(job_id: str):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("document_embeddings")

    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Only get entries for this job
    a_entries = collection.get(
        where={"$and": [{"index_type": "document_a"}, {"file_id": job_id}]}
    )

    print("Total entries found", len(a_entries["ids"]))

    output_path = f"comparison_{job_id}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Differences for Job {job_id}\n\n")

    for doc_a, meta_a, id_a in zip(a_entries["documents"], a_entries["metadatas"], a_entries["ids"]):
        print(f"Comparing {id_a}...")

        embedding = embed_text(doc_a)

        # Query Document B for this job
        result = collection.query(
            query_embeddings=embedding,
            where={"$and": [{"index_type": "document_b"}, {"file_id": job_id}]},
            n_results=1
        )

        if not result["documents"] or not result["documents"][0]:
            print(f"No match found for {id_a}")
            continue

        doc_b = result["documents"][0][0]
        meta_b = result["metadatas"][0][0]

        # GPT diff
        prompt = f"""You are an analyst comparing two versions of content.

Compare the following two sections and summarize the differences in plain English. Highlight key changes.

If they are functionally the same, say: "No significant difference."

**Document A (page {meta_a.get('pages')}):**
{doc_a}

**Document B (page {meta_b.get('pages')}):**
{doc_b}"""

        response = llm.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2
        )

        diff = response.choices[0].message.content.strip()

        if "no significant difference" in diff.lower():
            print(f"Omitting {id_a} - no significant difference.")
            continue

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"""## Difference for Page {meta_a.get('pages')}

{diff}

---
""")

    print(f"Markdown diff report saved to: {output_path}")

