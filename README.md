# PDF RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for asking questions over your own PDF documents. Upload a PDF, and ask natural-language questions about its contents — answers are generated using only the retrieved, relevant sections of the document.

Runs entirely on local models via [Ollama](https://ollama.com) — no external API keys or cloud LLM calls required.

## How it works

1. **Ingest** — A PDF is uploaded through the Streamlit UI. It's parsed and split into overlapping chunks using LlamaIndex's `SentenceSplitter`.
2. **Embed & store** — Each chunk is embedded locally using Ollama's `nomic-embed-text` model and stored in a [Qdrant](https://qdrant.tech) vector collection.
3. **Query** — When a question is asked, it's embedded the same way, and the top-k most similar chunks are retrieved from Qdrant.
4. **Generate** — The retrieved chunks are passed as context to Ollama's `llama3` model, which generates a grounded answer citing only the retrieved content.

Ingestion and querying are implemented as asynchronous, durable functions orchestrated by [Inngest](https://www.inngest.com/) and served through FastAPI, rather than as simple synchronous request handlers. This makes each pipeline step (chunking, embedding, upserting, retrieval, generation) independently retryable and observable.

## Tech Stack

| Component        | Tool                              |
|-------------------|------------------------------------|
| PDF parsing/chunking | LlamaIndex (`SentenceSplitter`) |
| Embeddings         | Ollama — `nomic-embed-text`      |
| Vector database    | Qdrant                           |
| LLM generation      | Ollama — `llama3`                |
| Orchestration       | Inngest + FastAPI                |
| Frontend            | Streamlit                        |
| Data validation     | Pydantic                         |

## Prerequisites

Before running the app, make sure the following are installed and running:

1. **Python 3.11+** (the `pyproject.toml` currently specifies `>=3.14` — double check this is intentional before pushing, as 3.14 may not be released/stable at the time of setup)
2. **[uv](https://docs.astral.sh/uv/)** for dependency management
3. **[Ollama](https://ollama.com/download)** installed and running locally, with the required models pulled:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3
   ```
4. **Qdrant**, running locally (easiest via Docker):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
5. **Inngest Dev Server**, for local orchestration:
   ```bash
   npx inngest-cli@latest dev
   ```

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd langchain3

# Install dependencies
uv sync

# Copy environment variables (if using any, e.g. for Inngest event keys)
cp .env.example .env  # create this if you have any env vars to configure
```

## Running the app

You'll need three processes running simultaneously, each in its own terminal:

**Terminal 1 — Qdrant** (if not already running via Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Terminal 2 — Inngest Dev Server**
```bash
npx inngest-cli@latest dev
```

**Terminal 3 — FastAPI app** (serves the Inngest functions)
```bash
uv run uvicorn main:app --reload
```

**Terminal 4 — Streamlit UI**
```bash
uv run streamlit run streamlit_app.py
```

Then open the Streamlit URL shown in your terminal (typically `http://localhost:8501`).

## Usage

1. Upload a PDF using the file uploader. This triggers an ingestion event — the PDF is chunked, embedded, and stored in Qdrant.
2. Once ingestion completes, type a question in the "Ask a question about your PDFs" section.
3. The app retrieves the most relevant chunks and streams back a generated answer, along with the source document(s) used.

## Project Structure

```
.
├── main.py              # FastAPI app + Inngest function definitions (ingest & query)
├── streamlit_app.py      # Streamlit frontend for upload and Q&A
├── data_loader.py         # PDF loading, chunking, and embedding logic
├── vector_db.py           # Qdrant client wrapper (create, upsert, search)
├── custom_types.py         # Pydantic models for typed data passed between steps
├── pyproject.toml           # Project dependencies
└── uploads/                  # Uploaded PDFs are saved here at runtime
```

## Known limitations / notes

- Embeddings are currently generated one chunk at a time (`BATCH_SIZE = 1` in `data_loader.py`) rather than batched, which is slower but was chosen for reliability with the local Ollama embedding endpoint.
- The app currently defaults to a single hardcoded Qdrant collection (`docs_ollama`); ingesting a differently-sized embedding model will trigger an automatic collection reset (see `vector_db.py`).
- Requires Ollama, Qdrant, and the Inngest dev server all running locally — there is currently no single-command startup script.

## Possible next steps

- Add a `docker-compose.yml` to spin up Qdrant + the app together.
- Batch embedding calls for faster ingestion of large PDFs.
- Add automated evaluation of retrieval quality (e.g., a small labeled Q&A set).
- Support additional file types beyond PDF (e.g., `.txt`, `.docx`).
