"""Singleton client لـ ChromaDB — يُستخدم من جميع الموديولات لتفادي تعارض الإعدادات."""

from __future__ import annotations

import os

_client = None
_client_path: str | None = None


def get_client(persist_directory: str | None = None):
    """إرجاع نفس instance من ChromaDB PersistentClient طوال عمر العملية.
    
    يستخدم متغير module-level بدلاً من lru_cache لضمان shared state حقيقي
    بين جميع الموديولات في نفس Python process.
    """
    global _client, _client_path

    path = persist_directory or os.environ.get("CHROMA_DB_PATH", "./chroma_db")

    # نفس الـ path → نفس الـ client
    if _client is not None and _client_path == path:
        return _client

    import chromadb
    from chromadb.config import Settings

    # allow_reset=True ثابتة دائماً لتفادي التعارض
    settings = Settings(allow_reset=True, anonymized_telemetry=False)

    try:
        _client = chromadb.PersistentClient(path=path, settings=settings)
    except Exception:
        # لو فيه instance قديم بـ settings مختلفة، نحاول نعمل reset
        chromadb.PersistentClient(path=path, settings=Settings(allow_reset=True, anonymized_telemetry=False))
        _client = chromadb.PersistentClient(path=path, settings=settings)

    _client_path = path
    return _client


def reset_client() -> None:
    """إعادة تعيين الـ client (مفيد بعد تغيير الـ path)."""
    global _client, _client_path
    _client = None
    _client_path = None
