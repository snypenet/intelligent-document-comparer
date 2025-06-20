# Intelligent Document Comparer

This project compares two similar PDF documents using OpenAI and ChromaDB to highlight meaningful differences in content. It is optimized for regulatory, pricing, or policy documents that may have minor edits or updates.

## Features

- Extracts pages from two input PDFs
- Chunks and stores document content with metadata for post-processing
- Generates embeddings for content using OpenAI's `text-embedding-3-small`
- Stores all data in a persistent ChromaDB instance on disk
- Skips re-processing if documents were previously analyzed
- Performs semantic nearest-neighbor matching of Document A pages against Document B
- Uses GPT-4 to compare matching content and outputs a Markdown diff report
- Ignores boilerplate or unchanged content

---

## Requirements

- Python 3.10+
- OpenAI API key (set as environment variable)
- Virtual environment with required dependencies (`requirements.txt`)

---

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/snypenet/intelligent-document-comparer.git
   cd intelligent-document-comparer
   ```

2. Set up your virtual environment:
   ```powershell
   .\Init-Environment.ps1  # PowerShell script to create/recreate venv and install dependencies
   ```

3. Create a `.env` file:
   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   ```

4. Run the script with two PDF files:
   ```bash
   python main.py --doc_a "path/to/document_a.pdf" --doc_b "path/to/document_b.pdf"
   ```

   You can also specify different search modes for comparison:
   ```bash
   # Topic-only search (default)
   python main.py --doc_a "path/to/document_a.pdf" --doc_b "path/to/document_b.pdf" --search-mode topic_only
   
   # Hybrid search (topic filtering + embedding similarity)
   python main.py --doc_a "path/to/document_a.pdf" --doc_b "path/to/document_b.pdf" --search-mode hybrid
   
   # Topic-aware semantic search (semantic search with topic bonus)
   python main.py --doc_a "path/to/document_a.pdf" --doc_b "path/to/document_b.pdf" --search-mode topic_aware
   ```

---

## Workflow

The document processing now follows a two-phase approach:

### Phase 1: Chunking and Storage
- Documents are chunked into meaningful sections
- Chunks are saved to working directories with metadata
- Topics are extracted and refined using LLM
- No embeddings are generated at this stage

### Phase 2: Embedding and Database Storage
- Chunks are read from working directories
- **LLM assigns relevant topics to each chunk** from the available topic list
- Embeddings are generated for each chunk
- Data is stored in ChromaDB for retrieval with assigned topics

### Phase 3: Comparison with Multiple Search Modes
- **Topic-Only Search**: Pure topic-based filtering for precise topic matching
- **Hybrid Search**: Combines topic filtering with embedding similarity for balanced results
- **Topic-Aware Search**: Semantic search with topic relevance bonus for comprehensive coverage

This separation allows for:
- Better error handling and recovery
- Ability to reprocess chunks without re-chunking
- Parallel processing of multiple documents
- Inspection of chunks before embedding
- **Topic-based filtering and analysis of chunks**
- **Multiple search strategies for different use cases**

---

## Utility Scripts

### Process Chunks Separately

If you want to process chunks from existing directories without re-chunking:

```bash
# List directories and their chunk counts
python process_chunks.py --dirs "runs/uuid/doc1" "runs/uuid/doc2" --list-only

# Process chunks and generate embeddings
python process_chunks.py --dirs "runs/uuid/doc1" "runs/uuid/doc2"
```

### Test Topic Assignment

Test the topic assignment functionality with sample data:

```bash
python test_topic_assignment.py
```

### Test Topic-Based Comparison

Test the topic-based querying and comparison functionality:

```bash
python test_topic_comparison.py
```

### Test Search Modes

Test all three search modes (topic_only, hybrid, topic_aware):

```bash
python test_search_modes.py
```

### Query by Topics

You can query chunks by their assigned topics using the database module:

```python
from pipeline.db import query_by_topics

# Query chunks that have "Payment Rates" or "Coverage Policies" assigned
results = query_by_topics(
    topics=["Payment Rates", "Coverage Policies"],
    n_results=20,
    where={"index_type": "document_a"}
)
```

### Hybrid Search

Combine topic filtering with embedding similarity:

```python
from pipeline.db import hybrid_search

# Hybrid search with topic filtering and semantic similarity
results = hybrid_search(
    query_text="Payment Rates",
    topics=["Payment Rates"],
    n_results=10,
    topic_weight=0.4,  # 40% weight for topic relevance
    embedding_weight=0.6  # 60% weight for semantic similarity
)
```

### Topic-Aware Semantic Search

Semantic search with topic relevance bonus:

```python
from pipeline.db import topic_aware_semantic_search

# Semantic search that boosts results with relevant topics
results = topic_aware_semantic_search(
    query_text="Payment Rates",
    topics=["Payment Rates"],
    n_results=10,
    topic_threshold=0.5  # Minimum topic relevance for bonus
)
```

---

## Output

A Markdown file named like this will be created:

```
comparison_<job_hash>.md
```

Only pages with **significant differences** are included. Each section contains a plain-English explanation of the change.

Example:
```markdown
## Difference for Page 2

The key difference between Document A and Document B is the amount assigned to CPT Code 99213...
```

---

## Data Storage

- Uses **ChromaDB PersistentClient** to store and reuse previously indexed documents.
- Data is persisted to `./chroma_db/`.
- Working directories are created in `./runs/` with unique UUIDs for each processing run.

---

## How It Works

1. **Job ID**: A short hash is generated from the filenames of Document A and B.
2. **Chunking Phase**:
   - Document A: Chunks pages and saves with metadata
   - Document B: Chunks pages with sliding window and saves with metadata
3. **Embedding Phase**: Generates embeddings for all chunks and stores in ChromaDB
4. **Topic Assignment**: LLM assigns relevant topics to each chunk during processing
5. **Topic-Based Comparison**: Each topic is compared by finding chunks that have been assigned that specific topic
6. **GPT-4 Comparison**: Differences are summarized in markdown format using OpenAI's GPT.

---

## TODO / Ideas

- Support batch comparisons
- Generate HTML or PDF reports
- Integrate into a Flask/Streamlit web UI
- Customize prompts or diff format

---

## Dependencies

See `requirements.txt` for details. Main packages used:

- `openai`
- `chromadb`
- `PyMuPDF` (`fitz`)
- `python-dotenv`