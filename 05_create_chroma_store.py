"""إنشاء قاعدة بيانات متجهات ChromaDB مع حفظ metadata والـ embeddings."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _get_existing_embedding_dimension(client, collection_name: str) -> int | None:
    """الحصول على بعد الـ embedding للـ collection الموجودة، أو None إذا لم توجد."""
    try:
        collection = client.get_collection(name=collection_name)
        if collection.count() == 0:
            return None
        peek = collection.peek(limit=1)
        embeddings = peek.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            first = embeddings[0]
            if hasattr(first, "__len__"):
                return len(first)
    except Exception:
        pass
    return None


def create_or_update_chroma_store(
    chunks: list[dict[str, Any]],
    persist_directory: str = "./chroma_db",
    collection_name: str = "law_rag",
    rebuild: bool = False,
) -> dict[str, Any]:
    """أنشئ أو حدّث collection داخل ChromaDB عبر ChromaManager الموحّد."""
    from core.chroma_manager import (
        delete_collection,
        get_chroma_client,
        get_or_create_collection,
    )

    client = get_chroma_client(persist_directory)

    if rebuild:
        delete_collection(collection_name, persist_directory)

    has_valid_embeddings = bool(chunks) and all(
        isinstance(chunk.get("embedding"), list) and len(chunk.get("embedding") or []) > 0
        for chunk in chunks
    )

    new_dim = None
    if has_valid_embeddings and chunks:
        new_dim = len(chunks[0].get("embedding") or [])

    existing_dim = None
    collection_exists = False
    if has_valid_embeddings and not rebuild:
        try:
            existing_dim = _get_existing_embedding_dimension(client, collection_name)
            collection_exists = existing_dim is not None
        except Exception:
            pass

        if collection_exists and existing_dim != new_dim:
            delete_collection(collection_name, persist_directory)
            collection_exists = False

    if has_valid_embeddings:
        collection = client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )
    else:
        collection = get_or_create_collection(collection_name, persist_directory)

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    start_index = collection.count() if collection_exists and not rebuild else 0

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
        ids.append(str(start_index + index))
        documents.append(text)
        embeddings.append(embedding)
        metadatas.append(metadata)

    if documents:
        if has_valid_embeddings:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return {
        "collection_name": collection_name,
        "persist_directory": persist_directory,
        "count": len(documents),
    }


if __name__ == "__main__":
    sample = [
        {
            "text": "المادة الأولى: يبدأ العمل من تاريخ التعيين.",
            "book": "الأول",
            "chapter": "الأول",
            "article_number": "1",
            "chunk_id": "chunk_1",
            "source_file": "demo.txt",
        }
    ]
    result = create_or_update_chroma_store(sample, persist_directory="./chroma_db", rebuild=True)
    print(result)
