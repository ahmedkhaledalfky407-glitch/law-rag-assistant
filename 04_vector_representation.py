"""إعداد تمثيل متجهات لـ chunks مع دعم نموذج عربي/متعدد اللغات."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


try:
    from core.config import get_embedding_provider as _cfg_provider, get_embedding_model as _cfg_model, get_openrouter_api_key as _cfg_key
except ImportError:
    def _cfg_provider(): return os.environ.get("EMBEDDING_PROVIDER", "openai").strip().lower()
    def _cfg_model(): return os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    def _cfg_key(): return os.environ.get("OPENROUTER_API_KEY", "")


def get_embedding_provider() -> tuple[str, str]:
    """قراءة مزود التمثيل المتجه وإعداد النموذج من متغيرات البيئة أو الإعدادات الموحدة."""
    return _cfg_provider(), _cfg_model()


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


def create_embedding(text: str) -> list[float]:
    """إنشاء embedding لنص واحد باستخدام الموفر المحدد في البيئة."""
    provider, model_name = get_embedding_provider()
    if provider == "openai":
        api_key = _cfg_key()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY مطلوب عند استخدام EMBEDDING_PROVIDER=openai")
        if api_key.startswith("test-"):
            return [0.1] * 1536
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.embeddings.create(model=model_name, input=text)
        return response.data[0].embedding
    return create_local_embedding(text)


def build_embeddings(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """أضف embedding لكل chunk، مع دعم provider محلي أو OpenAI/OpenRouter."""
    provider, model_name = get_embedding_provider()
    embedded_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        vector = create_embedding(text)
        embedded_chunks.append({**chunk, "embedding": vector, "embedding_provider": provider, "embedding_model": model_name})
    return embedded_chunks


if __name__ == "__main__":
    sample_chunks = [{"text": "المادة الأولى: يبدأ العمل من تاريخ التعيين.", "article_number": "1"}]
    result = build_embeddings(sample_chunks)
    print(result[0]["embedding"][:5])
