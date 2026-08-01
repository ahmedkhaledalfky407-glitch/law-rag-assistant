"""استرجاع أقرب السياقات القانونية من ChromaDB."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=32)
def _get_chroma_client(persist_directory: str, allow_reset: bool = True, anonymized_telemetry: bool = False):
    """إرجاع نفس العميل لـ ChromaDB داخل العملية لتفادي تعارض الإعدادات."""
    import chromadb
    from chromadb.config import Settings

    settings = Settings(allow_reset=allow_reset, anonymized_telemetry=anonymized_telemetry)
    return chromadb.PersistentClient(path=persist_directory, settings=settings)


def _get_collection(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """استرجاع النتائج من ChromaDB أو إرجاع قائمة فارغة عند الفشل."""
    try:
        persist_directory = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
        client = _get_chroma_client(persist_directory, allow_reset=True, anonymized_telemetry=False)
    except Exception:
        return []
    collection = client.get_collection(name=os.environ.get("CHROMA_COLLECTION", "law_rag"))
    results = collection.query(query_texts=[query], n_results=top_k)
    hits: list[dict[str, Any]] = []
    for index, document in enumerate(results.get("documents", [[]])[0]):
        metadata = results.get("metadatas", [[]])[0][index] if results.get("metadatas") else {}
        distance = results.get("distances", [[]])[0][index] if results.get("distances") else None
        hits.append(
            {
                "text": document,
                "source": metadata,
                "similarity": 1.0 - float(distance) if distance is not None else 0.0,
            }
        )
    return hits


def retrieve_context(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """إرجاع السياق الأكثر صلة بالسؤال، مع إعادة ترتيب بسيطة بناءً على ذكر المادة."""
    results = _get_collection(query, top_k=top_k)
    if not results:
        return []

    article_match = None
    match = __import__("re").search(r"المادة\s*(?:رقم\s*)?([\w\s]+)", query)
    if match:
        article_match = match.group(1).strip()

    if article_match:
        ranked = sorted(
            results,
            key=lambda item: (
                1 if str(item.get("source", {}).get("article_number", "")).strip() == article_match else 0,
                item.get("similarity", 0),
            ),
            reverse=True,
        )
        return ranked
    return results


if __name__ == "__main__":
    results = retrieve_context("ما هي المادة 1؟")
    print(results)
