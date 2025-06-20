import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("document_embeddings")

def store_embedding(doc: dict):
    metadata = {
        "file_id": doc["file_id"],
        "pages": doc.get("pages"),
        "summary": doc.get("summary"),
        "index_type": doc["index_type"],
        "assigned_topics": doc.get("assigned_topics", "")
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

def query_by_topics(topics: list[str], n_results: int = 10, where: dict = None):
    """
    Query chunks by assigned topics.
    
    Args:
        topics: List of topics to search for
        n_results: Number of results to return
        where: Additional metadata filters
        
    Returns:
        Query results from ChromaDB
    """
    # Since ChromaDB doesn't support $contains, we'll use a different approach
    # We'll query all chunks and filter by topics in Python
    combined_where = where
    
    # Debug logging
    print(f"[DEBUG] query_by_topics - topics: {topics}")
    print(f"[DEBUG] query_by_topics - where: {where}")
    print(f"[DEBUG] query_by_topics - combined_where: {combined_where}")
    
    # Get more results since we'll filter in Python
    result = collection.query(
        query_embeddings=None,  # We're filtering by metadata only
        n_results=n_results * 5,  # Get more results to filter from
        where=combined_where
    )
    
    if not result["documents"] or not result["documents"][0]:
        return result
    
    # Filter by topics in Python
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    ids = result["ids"][0]
    
    # Filter chunks that have any of the specified topics
    filtered_docs = []
    filtered_metas = []
    filtered_ids = []
    
    for doc, meta, doc_id in zip(docs, metas, ids):
        assigned_topics = meta.get("assigned_topics", "")
        if assigned_topics:
            # Split the pipe-separated topics
            chunk_topics = assigned_topics.split("||||") if assigned_topics else []
            # Check if any of the query topics match the chunk topics
            if any(topic in chunk_topics for topic in topics):
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_ids.append(doc_id)
        
        # Stop if we have enough results
        if len(filtered_docs) >= n_results:
            break
    
    return {
        "documents": [filtered_docs],
        "metadatas": [filtered_metas],
        "ids": [filtered_ids]
    }

def hybrid_search(query_text: str, topics: list[str] = None, n_results: int = 10, where: dict = None, 
                  topic_weight: float = 0.3, embedding_weight: float = 0.7):
    """
    Hybrid search that combines topic filtering with embedding similarity.
    
    Args:
        query_text: Text to search for (will be embedded)
        topics: List of topics to filter by (optional)
        n_results: Number of results to return
        where: Additional metadata filters
        topic_weight: Weight for topic-based filtering (0.0 to 1.0)
        embedding_weight: Weight for embedding similarity (0.0 to 1.0)
        
    Returns:
        Query results from ChromaDB with combined scoring
    """
    from .embed import embed_text
    
    # Generate embedding for query text
    query_embedding = embed_text(query_text)
    
    # Build where clause (without topic filtering since we'll do it in Python)
    combined_where = where
    
    # Debug logging
    print(f"[DEBUG] hybrid_search - query_text: {query_text}")
    print(f"[DEBUG] hybrid_search - topics: {topics}")
    print(f"[DEBUG] hybrid_search - where: {where}")
    print(f"[DEBUG] hybrid_search - combined_where: {combined_where}")
    
    # Perform the query (get more results since we'll filter and re-rank)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 5,  # Get more results for filtering and re-ranking
        where=combined_where
    )
    
    if not result["documents"] or not result["documents"][0]:
        return result
    
    # Extract results
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    ids = result["ids"][0]
    distances = result["distances"][0] if "distances" in result else [0.0] * len(docs)
    
    # Filter by topics if specified
    if topics:
        filtered_docs = []
        filtered_metas = []
        filtered_ids = []
        filtered_distances = []
        
        for doc, meta, doc_id, distance in zip(docs, metas, ids, distances):
            assigned_topics = meta.get("assigned_topics", "")
            if assigned_topics:
                # Split the pipe-separated topics
                chunk_topics = assigned_topics.split("||||") if assigned_topics else []
                # Check if any of the query topics match the chunk topics
                if any(topic in chunk_topics for topic in topics):
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
                    filtered_ids.append(doc_id)
                    filtered_distances.append(distance)
        
        # Use filtered results
        docs = filtered_docs
        metas = filtered_metas
        ids = filtered_ids
        distances = filtered_distances
    
    # Calculate hybrid scores
    hybrid_scores = []
    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances)):
        # Normalize embedding distance (lower is better, so invert)
        embedding_score = 1.0 - min(distance, 1.0) if distance is not None else 0.5
        
        # Calculate topic relevance score
        topic_score = 0.0
        if topics and meta.get("assigned_topics"):
            assigned_topics = meta["assigned_topics"].split("||||") if meta["assigned_topics"] else []
            # Count how many query topics match assigned topics
            matches = sum(1 for topic in topics if topic in assigned_topics)
            topic_score = matches / len(topics) if topics else 0.0
        
        # Combine scores with weights
        hybrid_score = (embedding_weight * embedding_score) + (topic_weight * topic_score)
        hybrid_scores.append(hybrid_score)
    
    # Sort by hybrid score (descending)
    sorted_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)
    
    # Return top n_results
    top_n = min(n_results, len(sorted_indices))
    sorted_docs = [docs[i] for i in sorted_indices[:top_n]]
    sorted_metas = [metas[i] for i in sorted_indices[:top_n]]
    sorted_ids = [ids[i] for i in sorted_indices[:top_n]]
    sorted_scores = [hybrid_scores[i] for i in sorted_indices[:top_n]]
    
    return {
        "documents": [sorted_docs],
        "metadatas": [sorted_metas],
        "ids": [sorted_ids],
        "distances": [sorted_scores]  # Return hybrid scores as distances
    }

def topic_aware_semantic_search(query_text: str, topics: list[str] = None, n_results: int = 10, 
                               where: dict = None, topic_threshold: float = 0.5):
    """
    Semantic search with topic awareness - prioritizes chunks that have relevant topics assigned.
    
    Args:
        query_text: Text to search for (will be embedded)
        topics: List of topics to consider for relevance scoring
        n_results: Number of results to return
        where: Additional metadata filters
        topic_threshold: Minimum topic relevance score to boost ranking
        
    Returns:
        Query results from ChromaDB with topic-aware scoring
    """
    from .embed import embed_text
    
    # Generate embedding for query text
    query_embedding = embed_text(query_text)
    
    # Perform semantic search
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 2,  # Get more results for re-ranking
        where=where
    )
    
    if not result["documents"] or not result["documents"][0]:
        return result
    
    # Extract results
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    ids = result["ids"][0]
    distances = result["distances"][0] if "distances" in result else [0.0] * len(docs)
    
    # Calculate topic-aware scores
    topic_aware_scores = []
    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances)):
        # Base semantic score (inverted distance)
        semantic_score = 1.0 - min(distance, 1.0) if distance is not None else 0.5
        
        # Topic relevance bonus
        topic_bonus = 0.0
        if topics and meta.get("assigned_topics"):
            assigned_topics = meta["assigned_topics"].split("||||") if meta["assigned_topics"] else []
            # Count how many query topics match assigned topics
            matches = sum(1 for topic in topics if topic in assigned_topics)
            topic_relevance = matches / len(topics) if topics else 0.0
            
            # Apply bonus if topic relevance is above threshold
            if topic_relevance >= topic_threshold:
                topic_bonus = topic_relevance * 0.3  # 30% bonus for topic relevance
        
        # Combine scores
        final_score = semantic_score + topic_bonus
        topic_aware_scores.append(final_score)
    
    # Sort by topic-aware score (descending)
    sorted_indices = sorted(range(len(topic_aware_scores)), key=lambda i: topic_aware_scores[i], reverse=True)
    
    # Return top n_results
    top_n = min(n_results, len(sorted_indices))
    sorted_docs = [docs[i] for i in sorted_indices[:top_n]]
    sorted_metas = [metas[i] for i in sorted_indices[:top_n]]
    sorted_ids = [ids[i] for i in sorted_indices[:top_n]]
    sorted_scores = [topic_aware_scores[i] for i in sorted_indices[:top_n]]
    
    return {
        "documents": [sorted_docs],
        "metadatas": [sorted_metas],
        "ids": [sorted_ids],
        "distances": [sorted_scores]  # Return topic-aware scores as distances
    }

