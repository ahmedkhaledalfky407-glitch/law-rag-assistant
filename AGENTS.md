# AGENTS.md — Project Knowledge

## Project: Egyptian Labor Law RAG Assistant

A Streamlit-based RAG (Retrieval-Augmented Generation) application for answering questions about **Egyptian Labor Law No. 14 of 2025**.

### Architecture

7 pipeline modules (loaded dynamically via `importlib.util` with numeric prefixes):

| File | Role |
|------|------|
| `01_documents.py` | Loads `.txt` files, returns raw text with metadata |
| `02_preprocessing.py` | Cleans Arabic text, extracts hierarchy (book/chapter/article) via regex |
| `03_chunking.py` | Splits long articles into overlapping chunks (max 800 tokens) |
| `04_vector_representation.py` | Creates embeddings — local (SHA-256 fallback) or OpenRouter |
| `05_create_chroma_store.py` | Upserts chunks into ChromaDB via singleton manager |
| `06_retrieve_context.py` | Queries ChromaDB, re-ranks by article number match |
| `07_prompting.py` | Builds system prompt, calls OpenRouter API via OpenAI SDK |
| `streamlit_app.py` | Main Streamlit UI (RTL Arabic, sidebar config, file upload, chat) |

### Key Files

- `core/chroma_manager.py` — ChromaDB singleton manager (prevents "different settings" errors in Streamlit reruns)
- `chroma_client.py` — Compatibility shim, delegates to `core.chroma_manager`
- `build_index.py` — CLI script to build the ChromaDB index from `data/` files
- `.env` — Environment variables (API key, model, ChromaDB path)
- `.env.example` — Template for `.env`
- `.streamlit/secrets.toml` — Streamlit Cloud secrets template

### Configuration

- **Embedding provider**: `local` (SHA-256 fallback, 16-dim vector) — no external model needed for basic operation
- **OpenRouter model**: `openai/gpt-4o-mini` (default)
- **ChromaDB**: persisted at `./chroma_db`, collection name `law_rag`

### Recent Changes

- **System prompt made flexible** (`07_prompting.py:58`): Changed from rigid refusal to conversational tone — handles greetings warmly, asks for clarification before answering, gently declines non-legal questions, and can answer general questions when no legal context is available. Restructured rules so greetings are handled first and the AI doesn't refuse immediately.