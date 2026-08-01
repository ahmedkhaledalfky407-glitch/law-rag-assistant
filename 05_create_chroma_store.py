"""إنشاء قاعدة بيانات متجهات ChromaDB مع حفظ metadata والـ embeddings."""

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


def create_or_update_chroma_store(chunks: list[dict[str, Any]], persist_directory: str = "./chroma_db", collection_name: str = "law_rag", rebuild: bool = False) -> dict[str, Any]:
    """أنشئ أو حدّث collection داخل ChromaDB."""
    try:
        client = _get_chroma_client(persist_directory, allow_reset=True, anonymized_telemetry=False)
    except Exception as exc:
        raise RuntimeError(f"ChromaDB غير متوفر: {exc}") from exc

    if rebuild:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    else:
        try:
            client.get_collection(name=collection_name)
        except Exception:
            pass

    has_valid_embeddings = bool(chunks) and all(
        isinstance(chunk.get("embedding"), list) and len(chunk.get("embedding") or []) > 0
        for chunk in chunks
    )

    if has_valid_embeddings:
        collection = client.get_or_create_collection(name=collection_name, embedding_function=None)
    else:
        collection = client.get_or_create_collection(name=collection_name)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        embedding = chunk.get("embedding") or []
        metadata = {
            "book": chunk.get("book") or "غير محدد",
            "chapter": chunk.get("chapter") or "غير محدد",
            "article_number": chunk.get("article_number") or "غير محدد",
            "chunk_id": chunk.get("chunk_id") or f"chunk_{index}",
            "source_file": chunk.get("source_file") or "unknown.txt",
        }
        ids.append(str(index))
        documents.append(text)
        embeddings.append(embedding)
        metadatas.append(metadata)

    if documents:
        if has_valid_embeddings:
            collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return {"collection_name": collection_name, "persist_directory": persist_directory, "count": len(documents)}


if __name__ == "__main__":
    sample_chunks = [{"text": "المادة الأولى: يبدأ العمل من تاريخ التعيين.", "embedding": [0.1, 0.2, 0.3], "book": "الأول", "chapter": "الأول", "article_number": "1", "chunk_id": "chunk_1", "source_file": "demo.txt"}]
    result = create_or_update_chroma_store(sample_chunks, persist_directory="./chroma_db", rebuild=True)
    print(result)
