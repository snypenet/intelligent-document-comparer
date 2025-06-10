import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("document_embeddings")

def store_embedding(doc: dict):
    metadata = {
        "file_id": doc["file_id"],
        "pages": doc.get("pages"),
        "summary": doc.get("summary"),
        "index_type": doc["index_type"]
    }

    metadata = {k: v for k, v in metadata.items() if v is not None}

    collection.add(
        documents=[doc["text"]],
        embeddings=[doc["embedding"]],
        ids=[doc["id"]],
        metadatas=[metadata]
    )

def exists(doc_id: str) -> bool:
    try:
        result = collection.get(ids=[doc_id])
        return len(result["ids"]) > 0
    except Exception:
        return False

