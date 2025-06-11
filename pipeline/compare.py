import os
import json
import chromadb
from openai import OpenAI
from .embed import embed_text


def compare_documents_to_markdown(job_id: str):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("document_embeddings")

    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Retrieve all document A chunks for the job
    a_entries = collection.get(where={"$and": [{"index_type": "document_a"}, {"file_id": job_id}]})

    output_path = f"comparison_{job_id}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Differences for Job {job_id}\n\n")

    for doc_a, meta_a, id_a in zip(a_entries["documents"], a_entries["metadatas"], a_entries["ids"]):
        print(f"Comparing {id_a}...")

        embedding = embed_text(doc_a)

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

        # LLM prompt to compare and extract subject + difference
        prompt = f"""You are a document comparison analyst.

Your job is to compare two chunks of text from different documents and determine if there are any **substantive, meaningful differences**. You should ignore formatting differences, repetition of content, or minor wording changes that do not affect the actual meaning.

If there is a meaningful difference, respond with a JSON object like:
{{
  "subject": "Short topic or subject line of the chunk",
  "difference": "Summarize the substantive change clearly and concisely"
}}

If there is **no meaningful difference**, respond with this exact JSON:
{{
  "subject": "NA",
  "difference": "No significant difference."
}}

**Rules:**
- Respond with valid JSON only — no markdown, no explanation, no commentary.
- Do NOT highlight differences in phrasing, wording, or layout unless it affects meaning.
- Do NOT call out duplicate or repeated content if the information is unchanged.
- If the content is semantically identical, you MUST return the "No significant difference" response.
- Inlude the page numbers from Document B in the differences

Now compare the chunks below.

Document A (pages {meta_a.get('pages')}):
{doc_a}

Document B (pages {meta_b.get('pages')}):
{doc_b}"""


        response = llm.chat.completions.create(
            model=os.getenv("MODEL_TYPE"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        try:
            parsed = json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"[Warning] Failed to parse response for {id_a}: {e}")
            continue

        subject = parsed.get("subject", "Untitled Topic")
        difference = parsed.get("difference", "No significant difference.")

        if "no significant difference" in difference.lower():
            print(f"Omitting {id_a} - no significant difference.")
            continue

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"## {subject} (Pages {meta_a.get('pages')})\n\n{difference}\n\n---\n")

    print(f"Markdown diff report saved to: {output_path}")
