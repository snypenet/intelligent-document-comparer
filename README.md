# Intelligent Document Comparer

This project compares two similar PDF documents using OpenAI and ChromaDB to highlight meaningful differences in content. It is optimized for regulatory, pricing, or policy documents that may have minor edits or updates.

## Features

- Extracts pages from two input PDFs
- Summarizes each page (for Document A)
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

---

## How It Works

1. **Job ID**: A short hash is generated from the filenames of Document A and B.
2. **Parallel Processing**:
   - Document A: Summarizes each page → embeds → stores.
   - Document B: Embeds page triplets → stores.
3. **Semantic Diff**: Each Document A page is matched to its closest Document B neighbor via ChromaDB.
4. **GPT-4 Comparison**: Differences are summarized in markdown format using OpenAI's GPT.

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