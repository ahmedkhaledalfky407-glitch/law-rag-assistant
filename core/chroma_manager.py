"""
ChromaDB Singleton Manager
==========================
مصدر واحد لـ PersistentClient في كامل التطبيق.

الحل الجذري لمشكلة "different settings":
- Settings يُنشأ مرة واحدة على مستوى الـ module
- Client يُنشأ مرة واحدة ويُخزَّن في module-level variable
- sys.modules يضمن عدم إعادة تحميل الـ module في Streamlit reruns
- إذا وُجد تعارض (مثلاً بعد rebuild)، نستخدم reset() لتنظيف الـ registry
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Settings الوحيد — يُنشأ عند أول import فقط
from chromadb.config import Settings as _S  # noqa: E402

_SETTINGS = _S(allow_reset=True, anonymized_telemetry=False)

_client = None
_client_path: str | None = None


def get_chroma_client(persist_directory: str | None = None):
    """إرجاع الـ ChromaDB client الوحيد في التطبيق."""
    global _client, _client_path

    path = persist_directory or os.environ.get("CHROMA_DB_PATH", "./chroma_db")

    # نفس الـ path ونفس الـ client موجود → ارجعه مباشرة
    if _client is not None and _client_path == path:
        return _client

    import chromadb

    try:
        _client = chromadb.PersistentClient(path=path, settings=_SETTINGS)
    except ValueError:
        # يوجد instance قديم بـ settings مختلفة → نعمل reset للـ registry
        logger.warning("ChromaManager: تعارض في الـ settings — إعادة تهيئة الـ registry")
        try:
            # نستخدم reset() على الـ client القديم إذا كان موجوداً
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient._identifier_to_system.clear()
        except Exception:
            pass
        _client = chromadb.PersistentClient(path=path, settings=_SETTINGS)

    _client_path = path
    logger.info("ChromaManager: client جاهز → %s", path)
    return _client


def get_collection(collection_name: str | None = None, persist_directory: str | None = None):
    """إرجاع collection من قاعدة البيانات."""
    name = collection_name or os.environ.get("CHROMA_COLLECTION", "law_rag")
    return get_chroma_client(persist_directory).get_collection(name=name)


def get_or_create_collection(
    collection_name: str | None = None,
    persist_directory: str | None = None,
    embedding_function=None,
):
    """إرجاع أو إنشاء collection."""
    name = collection_name or os.environ.get("CHROMA_COLLECTION", "law_rag")
    client = get_chroma_client(persist_directory)
    if embedding_function is None:
        return client.get_or_create_collection(name=name)
    return client.get_or_create_collection(name=name, embedding_function=embedding_function)


def delete_collection(collection_name: str, persist_directory: str | None = None) -> None:
    """حذف collection."""
    try:
        get_chroma_client(persist_directory).delete_collection(name=collection_name)
    except Exception:
        pass


def verify_singleton() -> dict:
    """حالة الـ singleton للتشخيص."""
    return {
        "module_id": id(sys.modules.get(__name__)),
        "client_id": id(_client) if _client else None,
        "settings_id": id(_SETTINGS),
        "client_path": _client_path,
        "client_ready": _client is not None,
        "collection_count": _client.get_collection(
            os.environ.get("CHROMA_COLLECTION", "law_rag")
        ).count() if _client else 0,
    }
