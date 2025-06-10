import os
from openai import OpenAI

def embed_text(text: str):
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text.strip()
    )
    return response.data[0].embedding
