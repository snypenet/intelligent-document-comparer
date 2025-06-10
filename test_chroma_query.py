import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Load the persistent client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("document_embeddings")

# Optional: print all stored IDs
print("Stored document IDs:")
print(collection.get()["ids"])

# Optional: perform a similarity query
query_text = "Medicare OPPS payment adjustment"
embedding_function = DefaultEmbeddingFunction()
query_embedding = embedding_function(query_text)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("\nTop matches:")
for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
    print("Summary or Text Snippet:")
    print(doc[:300])
    print("Metadata:", metadata)
    print("---")
