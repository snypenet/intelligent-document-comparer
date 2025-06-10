from openai import OpenAI
import os

def summarize_page(text: str) -> str:
    prompt = """You are an analyst extracting core content for vector search.
Given the following document page, extract only the meaningful changes or rules, ignoring any headers, footers, citations, or repeated information.

Output should be plain text, ≤ 400 tokens. Skip unrelated boilerplate."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt + "\n\n" + text}],
        max_tokens=400,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
