"""استرجاع أقرب السياقات القانونية من ChromaDB."""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _get_collection(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """استرجاع النتائج عبر ChromaManager الموحّد."""
    try:
        from core.chroma_manager import get_collection

        collection = get_collection(
            collection_name=os.environ.get("CHROMA_COLLECTION", "law_rag"),
            persist_directory=os.environ.get("CHROMA_DB_PATH", "./chroma_db"),
        )
    except Exception:
        return []

    try:
        results = collection.query(query_texts=[query], n_results=top_k)
    except Exception:
        return []

    hits: list[dict[str, Any]] = []
    for index, document in enumerate(results.get("documents", [[]])[0]):
        metadata = (
            results.get("metadatas", [[]])[0][index]
            if results.get("metadatas")
            else {}
        )
        distance = (
            results.get("distances", [[]])[0][index]
            if results.get("distances")
            else None
        )
        hits.append(
            {
                "text": document,
                "source": metadata,
                "similarity": 1.0 - float(distance) if distance is not None else 0.0,
            }
        )
    return hits


def retrieve_context(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """إرجاع السياق الأكثر صلة، مع إعادة ترتيب بسيطة بناءً على رقم المادة."""
    results = _get_collection(query, top_k=top_k)
    if not results:
        return []

    article_match: str | None = None
    match = re.search(r"المادة\s*(?:رقم\s*)?([\w\s]+)", query)
    if match:
        article_match = match.group(1).strip()

    if article_match:
        return sorted(
            results,
            key=lambda item: (
                1
                if str(item.get("source", {}).get("article_number", "")).strip()
                == article_match
                else 0,
                item.get("similarity", 0),
            ),
            reverse=True,
        )
    return results


if __name__ == "__main__":
    results = retrieve_context("ما هي المادة 1؟")
    print(results)
