"""استرجاع أقرب السياقات القانونية من ChromaDB."""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _get_collection(query: str, top_k: int = 10) -> list[dict[str, Any]]:
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


def _normalize_article_number(num: str) -> str:
    """تطبيع رقم المادة للمقارنة."""
    num = num.strip()
    arabic_numerals = "٠١٢٣٤٥٦٧٨٩"
    mapping = str.maketrans(arabic_numerals, "0123456789")
    num = num.translate(mapping)
    num = re.sub(r"^المادة\s*", "", num)
    num = re.sub(r"^رقم\s*", "", num)
    return num.strip()


def retrieve_context(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """إرجاع السياق الأكثر صلة، مع إعادة ترتيب وتنظيف النتائج."""
    results = _get_collection(query, top_k=top_k)
    if not results:
        return []

    article_match: str | None = None
    match = re.search(r"المادة\s*(?:رقم\s*)?([\w\s]+)", query)
    if match:
        article_match = _normalize_article_number(match.group(1))

    scored: list[dict[str, Any]] = []
    for item in results:
        source = item.get("source", {})
        article_num = source.get("article_number", "")
        normalized_article = _normalize_article_number(str(article_num))

        score = item.get("similarity", 0)

        if article_match and normalized_article == article_match:
            score += 0.5

        if article_match and normalized_article and normalized_article in article_match:
            score += 0.3

        if article_match and article_match in normalized_article:
            score += 0.3

        scored.append({**item, "score": score})

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)

    seen_articles: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in scored:
        article_num = str(item.get("source", {}).get("article_number", ""))
        if article_num not in seen_articles:
            seen_articles.add(article_num)
            deduped.append(item)

    return deduped


if __name__ == "__main__":
    results = retrieve_context("ما هي المادة 1؟")
    print(results)
