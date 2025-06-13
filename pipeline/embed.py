import os
from openai import OpenAI
from .utils import call_with_retries

def embed_text(text: str):
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = call_with_retries(lambda: client.embeddings.create(
        model="text-embedding-3-small",
        input=text.strip()
    ))
    return response.data[0].embedding
