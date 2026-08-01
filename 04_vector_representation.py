"""إعداد تمثيل متجهات لـ chunks مع دعم نموذج عربي/متعدد اللغات."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_embedding_provider() -> tuple[str, str]:
    """قراءة مزود التمثيل المتجه وإعداد النموذج من متغيرات البيئة."""
    provider = os.environ.get("EMBEDDING_PROVIDER", "local").strip().lower()
    model_name = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    return provider, model_name


def create_local_embedding(text: str) -> list[float]:
    """إنشاء embedding بسيط محلي كنسخة احتياطية عندما لا يتوفر نموذج ثقيل."""
    text = text.lower().strip()
    if not text:
        return [0.0] * 16
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    values: list[float] = []
    for i in range(16):
        chunk = digest[i % len(digest)]
        values.append(float(ord(chunk)) / 255.0)
    return values


def build_embeddings(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """أضف embedding لكل chunk، مع دعم local fallback دون الاعتماد على مفتاح API."""
    provider, model_name = get_embedding_provider()
    embedded_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        try:
            if provider == "openai" or os.environ.get("OPENROUTER_API_KEY"):
                from openai import OpenAI

                client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY", ""), base_url="https://openrouter.ai/api/v1")
                response = client.embeddings.create(model=model_name, input=text)
                vector = response.data[0].embedding
            else:
                vector = create_local_embedding(text)
        except Exception:
            vector = create_local_embedding(text)

        embedded_chunks.append({**chunk, "embedding": vector, "embedding_provider": provider, "embedding_model": model_name})
    return embedded_chunks


if __name__ == "__main__":
    sample_chunks = [{"text": "المادة الأولى: يبدأ العمل من تاريخ التعيين.", "article_number": "1"}]
    result = build_embeddings(sample_chunks)
    print(result[0]["embedding"][:5])
