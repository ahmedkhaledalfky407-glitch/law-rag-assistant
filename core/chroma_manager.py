"""
ChromaDB Singleton Manager
==========================
هذا الملف هو المصدر الوحيد لإنشاء ChromaDB client في كامل التطبيق.

القاعدة الصارمة:
    - لا يُسمح لأي ملف آخر باستدعاء PersistentClient(...) أو Settings(...)
    - كل الكود يستدعي get_chroma_client() من هذا الملف فقط

لماذا هذا يحل المشكلة:
    - ChromaDB's SharedSystemClient يقارن Settings objects بـ identity/equality
    - إذا تعددت Settings objects في نفس الـ process → ValueError
    - نحن ننشئ Settings مرة واحدة على مستوى الـ module (عند أول import)
    - sys.modules يضمن أن هذا الـ module يُحمَّل مرة واحدة فقط مهما تكرر import
    - Streamlit reruns تعيد exec الـ script لكن لا تعيد تحميل sys.modules
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# ============================================================
# الـ Settings الوحيد في التطبيق — يُنشأ هنا مرة واحدة فقط
# ============================================================
try:
    from chromadb.config import Settings as _ChromaSettings

    _CHROMA_SETTINGS: _ChromaSettings = _ChromaSettings(
        allow_reset=True,
        anonymized_telemetry=False,
    )
except ImportError as _import_err:
    raise RuntimeError(
        "chromadb غير مثبت. شغّل: pip install chromadb"
    ) from _import_err


# ============================================================
# الـ Client الوحيد في التطبيق — يُنشأ عند أول استدعاء
# ============================================================
_client = None
_client_path: str | None = None


class ChromaInitError(RuntimeError):
    """خطأ مخصص عند فشل تهيئة ChromaDB."""


def get_chroma_client(persist_directory: str | None = None):
    """
    إرجاع ChromaDB PersistentClient الوحيد في التطبيق.

    - أول استدعاء: ينشئ الـ client ويخزنه في module-level variable
    - الاستدعاءات التالية: يرجع نفس الـ object مباشرة
    - Streamlit reruns: sys.modules يحفظ الـ module → نفس الـ client دائماً
    - تغيير الـ path: يرجع client جديد ويخزنه (نادراً يحدث)
    """
    global _client, _client_path

    path = (
        persist_directory
        or os.environ.get("CHROMA_DB_PATH", "./chroma_db")
    )

    # نفس الـ path → نفس الـ client بدون أي عمل
    if _client is not None and _client_path == path:
        return _client

    import chromadb

    try:
        logger.info("ChromaManager: إنشاء PersistentClient → %s", path)
        _client = chromadb.PersistentClient(
            path=path,
            settings=_CHROMA_SETTINGS,
        )
        _client_path = path
        logger.info("ChromaManager: Client جاهز ✓")
    except ValueError as exc:
        # هذا يحدث فقط إذا خرقنا القاعدة وأنشأنا client آخر
        raise ChromaInitError(
            f"[ChromaManager] تعارض في الإعدادات! تأكد أن لا مكان آخر "
            f"يستدعي PersistentClient() أو Settings() مباشرة.\n"
            f"التفاصيل: {exc}"
        ) from exc
    except Exception as exc:
        raise ChromaInitError(
            f"[ChromaManager] فشل تهيئة ChromaDB: {exc}"
        ) from exc

    return _client


def get_collection(collection_name: str | None = None, persist_directory: str | None = None):
    """
    إرجاع collection مباشرة — shortcut مفيد لـ 06_retrieve_context.
    """
    name = collection_name or os.environ.get("CHROMA_COLLECTION", "law_rag")
    client = get_chroma_client(persist_directory)
    return client.get_collection(name=name)


def get_or_create_collection(
    collection_name: str | None = None,
    persist_directory: str | None = None,
    embedding_function=None,
):
    """
    إرجاع أو إنشاء collection — shortcut لـ 05_create_chroma_store.
    """
    name = collection_name or os.environ.get("CHROMA_COLLECTION", "law_rag")
    client = get_chroma_client(persist_directory)
    if embedding_function is None:
        return client.get_or_create_collection(name=name)
    return client.get_or_create_collection(
        name=name, embedding_function=embedding_function
    )


def delete_collection(collection_name: str, persist_directory: str | None = None) -> None:
    """حذف collection — يُستخدم عند rebuild فقط."""
    client = get_chroma_client(persist_directory)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # لو مش موجودة أصلاً، تجاهل


def verify_singleton() -> dict:
    """
    تحقق من حالة الـ singleton — للتشخيص فقط.
    """
    return {
        "module_id": id(sys.modules.get(__name__)),
        "client_id": id(_client) if _client else None,
        "settings_id": id(_CHROMA_SETTINGS),
        "client_path": _client_path,
        "client_ready": _client is not None,
    }
