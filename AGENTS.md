# AGENTS.md — Project Knowledge

## Project: Egyptian Labor Law RAG Assistant

A Streamlit-based RAG (Retrieval-Augmented Generation) application for answering questions about **Egyptian Labor Law No. 14 of 2025**.

### Architecture

7 pipeline modules (loaded dynamically via `importlib.util` with numeric prefixes):

| File | Role |
|------|------|
| `01_documents.py` | Loads `.txt` files, returns raw text with metadata |
| `02_preprocessing.py` | Cleans Arabic text, extracts hierarchy (book/chapter/article) via regex |
| `03_chunking.py` | Splits long articles into overlapping chunks (max 1200 tokens) |
| `04_vector_representation.py` | Creates embeddings via OpenRouter `openai/text-embedding-3-small` (1536-dim) |
| `05_create_chroma_store.py` | Upserts chunks into ChromaDB via singleton manager |
| `06_retrieve_context.py` | Queries ChromaDB with OpenRouter embeddings, re-ranks by article number match |
| `07_prompting.py` | Builds system prompt with [حرج قصوى] tags, calls OpenRouter API via OpenAI SDK |
| `streamlit_app.py` | Main Streamlit UI (RTL Arabic, sidebar config, file upload, chat) |

### Key Files

- `core/chroma_manager.py` — ChromaDB singleton manager (prevents "different settings" errors in Streamlit reruns)
- `chroma_client.py` — Compatibility shim, delegates to `core.chroma_manager`
- `build_index.py` — CLI script to build the ChromaDB index from `data/` files
- `.env` — Environment variables (API key, model, ChromaDB path, embedding provider)
- `.env.example` — Template for `.env`
- `.streamlit/secrets.toml` — Streamlit Cloud secrets template

### Configuration

- **Embedding provider**: `openai` (OpenRouter `openai/text-embedding-3-small`, 1536-dim semantic embeddings)
- **OpenRouter model**: `openai/gpt-4o-mini` (default)
- **ChromaDB**: persisted at `./chroma_db`, collection name `law_rag`

### Complete Audit Report

#### Bugs Found & Fixed

1. **SHA-256 local embeddings were random** (`.env`, `04_vector_representation.py`): `EMBEDDING_PROVIDER=local` used SHA-256 hash-based 16-dim vectors with zero semantic meaning. ChromaDB similarity search returned completely irrelevant chunks. **Fix**: Switched to `EMBEDDING_PROVIDER=openai` with `openai/text-embedding-3-small` (1536-dim semantic embeddings).

2. **Query embedding dimension mismatch** (`06_retrieve_context.py`): When querying ChromaDB with `query_texts`, ChromaDB used its default 384-dim embedding, but the collection stored 1536-dim OpenRouter embeddings, causing a dimension mismatch error. **Fix**: Added `_embed_query()` function that uses OpenRouter embeddings for queries to match stored vectors.

3. **System prompt too rigid** (`07_prompting.py`): Original prompt refused all non-legal questions with "هذا السؤال خارج نطاق تخصصي القانوني". **Fix**: Made prompt flexible — handles greetings warmly, asks for clarification, gently declines non-legal questions.

4. **Chat history not persisting** (`streamlit_app.py`): Only the latest message was displayed on each rerun. **Fix**: Added loop to re-render all messages from `st.session_state.messages`.

5. **Chunk size too small** (`03_chunking.py`): 800 tokens caused loss of legal context. **Fix**: Increased to 1200 tokens.

6. **Overlap ratio too low** (`03_chunking.py`): 0.12 caused discontinuity between chunks. **Fix**: Increased to 0.2.

7. **Retrieval top_k too small** (`06_retrieve_context.py`, `streamlit_app.py`): Only 5 results fetched. **Fix**: Increased to 10.

8. **No article number normalization** (`06_retrieve_context.py`, `02_preprocessing.py`): Arabic vs English numerals caused matching failures. **Fix**: Added `_normalize_article_number()` in both modules.

9. **No deduplication** (`06_retrieve_context.py`): Multiple chunks from same article cluttered results. **Fix**: Added deduplication by article number.

10. **Upload stages hidden** (`streamlit_app.py`): Only a generic spinner showed during upload. **Fix**: Added stage-by-stage progress tracking.

#### Root Cause of "No Source Found" Issue

The root cause was **#1** — SHA-256 local embeddings produced random vectors with no semantic meaning. When the user searched for "اجازات العمال", ChromaDB returned completely irrelevant chunks (e.g., articles about apprenticeship agreements), so the model had no relevant context and fell back to "no source found".

### Recent Changes

- **Added debug logging** (`06_retrieve_context.py`, `07_prompting.py`): Added `logging` module with INFO/WARNING/ERROR levels to all key steps in retrieval and LLM generation.
- **Complete audit performed**: All 10 bugs identified and fixed. System fully operational with OpenRouter semantic embeddings.