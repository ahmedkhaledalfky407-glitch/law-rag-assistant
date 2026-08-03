"""بناء فهرس ChromaDB من ملفات القانون في data/."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def load_module(module_name: str, file_name: str):
    module_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"تعذر تحميل الوحدة: {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    # تهيئة ChromaManager قبل أي شيء
    from core.chroma_manager import get_chroma_client, verify_singleton
    get_chroma_client()
    info = verify_singleton()
    print(f"ChromaManager جاهز: {info}")

    documents_module = load_module("documents", "01_documents.py")
    preprocessing_module = load_module("preprocessing", "02_preprocessing.py")
    chunking_module = load_module("chunking", "03_chunking.py")
    vector_module = load_module("vector_representation", "04_vector_representation.py")
    chroma_module = load_module("create_chroma_store", "05_create_chroma_store.py")

    data_dir = Path(__file__).with_name("data")
    text_files = sorted(data_dir.glob("*.txt"))
    print(f"الملفات المُعثر عليها: {[str(f) for f in text_files]}")

    documents = []
    for text_file in text_files:
        docs = documents_module.load_documents(str(text_file))
        documents.extend(docs)
    print(f"تم تحميل {len(documents)} وثيقة")

    articles = preprocessing_module.process_documents(documents)
    print(f"تم استخراج {len(articles)} مادة قانونية")

    chunks = chunking_module.chunk_articles(articles)
    print(f"تم إنشاء {len(chunks)} chunk")

    provider = os.environ.get("EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider == "local":
        embedded_chunks = chunks
        print(f"استخدام ChromaDB الافتراضي للـ embeddings ({len(embedded_chunks)} chunk)")
    else:
        embedded_chunks = vector_module.build_embeddings(chunks)
        print(f"تم إنشاء embeddings مخصّصة لـ {len(embedded_chunks)} chunk")

    result = chroma_module.create_or_update_chroma_store(
        embedded_chunks,
        persist_directory=os.environ.get("CHROMA_DB_PATH", "./chroma_db"),
        collection_name=os.environ.get("CHROMA_COLLECTION", "law_rag"),
        rebuild=True,
    )
    print(f"تم إنشاء ChromaDB: {result}")


if __name__ == "__main__":
    main()
