"""استرجاع أقرب السياقات القانونية من ChromaDB."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_ARABIC_NUMERALS = "٠١٢٣٤٥٦٧٨٩"

_ARABIC_UNITS = {
    "الأولى": 1,
    "الأولي": 1,
    "الثانية": 2,
    "الثالثة": 3,
    "الرابعة": 4,
    "الخامسة": 5,
    "السادسة": 6,
    "السابعة": 7,
    "الثامنة": 8,
    "التاسعة": 9,
    "العاشرة": 10,
}

_ARABIC_TEENS = {
    "الحادية عشرة": 11,
    "الحادية عشر": 11,
    "الثانية عشرة": 12,
    "الثانية عشر": 12,
    "الثالثة عشرة": 13,
    "الثالثة عشر": 13,
    "الرابعة عشرة": 14,
    "الرابعة عشر": 14,
    "الخامسة عشرة": 15,
    "الخامسة عشر": 15,
    "السادسة عشرة": 16,
    "السادسة عشر": 16,
    "السابعة عشرة": 17,
    "السابعة عشر": 17,
    "الثامنة عشرة": 18,
    "الثامنة عشر": 18,
    "التاسعة عشرة": 19,
    "التاسعة عشر": 19,
}

_ARABIC_TENS = {
    "العشرون": 20,
    "العشرين": 20,
    "الثلاثون": 30,
    "الثلاثين": 30,
    "الأربعون": 40,
    "الأربعين": 40,
    "الخمسون": 50,
    "الخمسين": 50,
    "الستون": 60,
    "الستين": 60,
    "السبعون": 70,
    "السبعين": 70,
    "الثمانون": 80,
    "الثمانين": 80,
    "التسعون": 90,
    "التسعين": 90,
}


def _arabic_word_to_int(text: str) -> str:
    """تحويل أرقام عربية مكتوبة بالكلمات إلى أرقام إنجليزية."""
    text = text.strip()
    if text in _ARABIC_UNITS:
        return str(_ARABIC_UNITS[text])
    if text in _ARABIC_TEENS:
        return str(_ARABIC_TEENS[text])
    if text in _ARABIC_TENS:
        return str(_ARABIC_TENS[text])
    for unit_name, unit_val in _ARABIC_UNITS.items():
        for ten_name, ten_val in _ARABIC_TENS.items():
            if text == f"{unit_name} و{ten_name}" or text == f"{unit_name} وال{ten_name}":
                return str(unit_val + ten_val)
    return text


def _embed_query(query: str) -> list[float]:
    """Embed query using OpenRouter if available, otherwise fallback."""
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning("_embed_query: no API key available")
            return []
        model = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.embeddings.create(model=model, input=query)
        logger.info("_embed_query: embedded query with dimension %d", len(response.data[0].embedding))
        return response.data[0].embedding
    except Exception as exc:
        logger.error("_embed_query: failed - %s", exc)
        return []


def _get_collection(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """استرجاع النتائج عبر ChromaManager الموحّد."""
    try:
        from core.chroma_manager import get_collection

        collection = get_collection(
            collection_name=os.environ.get("CHROMA_COLLECTION", "law_rag"),
            persist_directory=os.environ.get("CHROMA_DB_PATH", "./chroma_db"),
        )
        logger.info("_get_collection: collection='%s', count=%d", collection.name, collection.count())
    except Exception as exc:
        logger.error("_get_collection: failed to get collection - %s", exc)
        return []

    try:
        embedding = _embed_query(query)
        if embedding:
            results = collection.query(query_embeddings=[embedding], n_results=top_k)
            logger.info("_get_collection: used query_embeddings (dimension=%d)", len(embedding))
        else:
            existing_dim = None
            try:
                peek = collection.peek(limit=1)
                embeds = peek.get("embeddings")
                if embeds and len(embeds) > 0 and isinstance(embeds[0], list):
                    existing_dim = len(embeds[0])
            except Exception:
                pass

            if existing_dim is not None and existing_dim != 384:
                logger.error(
                    "_get_collection: cannot fall back to query_texts because collection uses external embeddings (dimension=%d)",
                    existing_dim,
                )
                return []

            results = collection.query(query_texts=[query], n_results=top_k)
            logger.warning("_get_collection: fell back to query_texts (no embedding)")
    except Exception as exc:
        logger.error("_get_collection: query failed - %s", exc)
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
    logger.info("_get_collection: returned %d hits", len(hits))
    return hits


def _normalize_article_number(num: str) -> str:
    """تطبيع رقم المادة للمقارنة."""
    num = num.strip()
    num = _arabic_word_to_int(num)
    arabic_numerals = "٠١٢٣٤٥٦٧٨٩"
    mapping = str.maketrans(arabic_numerals, "0123456789")
    num = num.translate(mapping)
    num = re.sub(r"^المادة\s*", "", num)
    num = re.sub(r"^رقم\s*", "", num)
    return num.strip()


def retrieve_context(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """إرجاع السياق الأكثر صلة، مع إعادة ترتيب وتنظيف النتائج."""
    logger.info("retrieve_context: query='%s', top_k=%d", query, top_k)
    results = _get_collection(query, top_k=top_k)
    if not results:
        logger.warning("retrieve_context: no results found")
        return []

    article_match: str | None = None
    match = re.search(r"المادة\s*(?:رقم\s*)?([\w\s]+)", query)
    if match:
        article_match = _normalize_article_number(match.group(1))
        logger.info("retrieve_context: article_match='%s'", article_match)

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

    logger.info("retrieve_context: returned %d deduped results", len(deduped))
    return deduped


if __name__ == "__main__":
    results = retrieve_context("ما هي المادة 1؟")
    print(results)
