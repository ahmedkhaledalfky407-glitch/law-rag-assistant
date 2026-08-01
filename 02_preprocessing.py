"""تنظيف النص العربي واستخراج البنية الهرمية للكتاب/الباب/المادة."""

from __future__ import annotations

import re
import sys
from typing import Any


def preprocess_text(text: str) -> str:
    """نظّف النص العربي مع الحفاظ على المعنى القانوني."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u202A\u202B\u202C\u202D\u202E]", "", text)
    text = re.sub(r"[\t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_hierarchy(raw_text: str) -> list[dict[str, Any]]:
    """استخراج عناصر {book, chapter, article_number, article_text} عبر Regex."""
    cleaned = preprocess_text(raw_text)
    articles: list[dict[str, Any]] = []
    current_book = None
    current_chapter = None

    for line in cleaned.splitlines():
        if not line.strip():
            continue
        book_match = re.search(r"الكتاب\s+([\w\s]+)", line)
        if book_match:
            current_book = book_match.group(1).strip()
            continue

        chapter_match = re.search(r"الباب\s+([\w\s]+)", line)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()
            continue

        article_match = re.search(r"المادة\s*(?:رقم\s*)?([\w\s]+)", line)
        if article_match:
            article_number = article_match.group(1).strip()
            article_text = line.replace(article_match.group(0), "", 1).strip()
            articles.append(
                {
                    "book": current_book,
                    "chapter": current_chapter,
                    "article_number": article_number,
                    "article_text": article_text,
                }
            )

    if not articles:
        articles.append(
            {
                "book": current_book,
                "chapter": current_chapter,
                "article_number": "غير محدد",
                "article_text": cleaned,
            }
        )
    return articles


def process_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """تحويل وثائق إلى قائمة من المواد المنظمة."""
    parsed_articles: list[dict[str, Any]] = []
    for document in documents:
        raw_text = document.get("raw_text", "")
        parsed_articles.extend(extract_hierarchy(raw_text))
    return parsed_articles


if __name__ == "__main__":
    sample_text = """الكتاب الأول\nالباب الأول\nالمادة رقم 1: يبدأ العمل من تاريخ التعيين.\nالمادة الثانية: تلتزم الجهات بأحكام هذا القانون.\n"""
    articles = extract_hierarchy(sample_text)
    print(f"تم استخراج {len(articles)} مادة")
    for article in articles:
        print(article)
