"""تقسيم المواد القانونية إلى chunks مع الحفاظ على السياق القانوني."""

from __future__ import annotations

import math
import re
import sys
from typing import Any


def split_text_by_sentences(text: str) -> list[str]:
    """تقسيم النص إلى جمل عربية باستخدام علامات الترقيم الشائعة."""
    parts = re.split(r"(?<=[.؛:])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_articles(articles: list[dict[str, Any]], max_tokens: int = 800, overlap_ratio: float = 0.12) -> list[dict[str, Any]]:
    """قسم كل مادة إلى chunks، مع تقسيم طويلها بشكل تكراري عند الحاجة."""
    chunks: list[dict[str, Any]] = []
    overlap_size = max(1, int(max_tokens * overlap_ratio))

    for index, article in enumerate(articles):
        text = article.get("article_text", "")
        if not text:
            continue
        estimated_tokens = max(1, len(text.split()))
        if estimated_tokens <= max_tokens:
            chunk_text = text
            chunks.append(
                {
                    "book": article.get("book"),
                    "chapter": article.get("chapter"),
                    "article_number": article.get("article_number"),
                    "chunk_id": f"{article.get('article_number')}_{index}_0",
                    "text": chunk_text,
                }
            )
            continue

        sentences = split_text_by_sentences(text)
        buffer: list[str] = []
        current_tokens = 0
        chunk_counter = 0
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens > max_tokens and buffer:
                chunk_text = " ".join(buffer)
                chunks.append(
                    {
                        "book": article.get("book"),
                        "chapter": article.get("chapter"),
                        "article_number": article.get("article_number"),
                        "chunk_id": f"{article.get('article_number')}_{index}_{chunk_counter}",
                        "text": chunk_text,
                    }
                )
                overlap_sentences = buffer[-overlap_size // max(1, len(buffer)):] if buffer else []
                buffer = list(overlap_sentences)
                current_tokens = sum(len(part.split()) for part in buffer)
                chunk_counter += 1
            buffer.append(sentence)
            current_tokens += sentence_tokens

        if buffer:
            chunks.append(
                {
                    "book": article.get("book"),
                    "chapter": article.get("chapter"),
                    "article_number": article.get("article_number"),
                    "chunk_id": f"{article.get('article_number')}_{index}_{chunk_counter}",
                    "text": " ".join(buffer),
                }
            )

    return chunks


if __name__ == "__main__":
    sample_articles = [
        {"book": "الأول", "chapter": "الأول", "article_number": "1", "article_text": "تبدأ أحكام هذا القانون من تاريخ العمل وتلتزم الجهات المختصة بتنفيذه على نحو يحقق العدالة."},
        {"book": "الأول", "chapter": "الأول", "article_number": "2", "article_text": "يُحدد هذا النص على هيئة مادة قانونية طويلة جدًا تتكرر فيها الجمل وتحتاج إلى تقسيم واضح حتى يحافظ السياق على معنى المادة القانونية عند الاسترجاع."},
    ]
    chunks = chunk_articles(sample_articles)
    print(f"تم إنشاء {len(chunks)} chunk")
    for chunk in chunks:
        print(chunk["chunk_id"], "->", chunk["text"][:80])
