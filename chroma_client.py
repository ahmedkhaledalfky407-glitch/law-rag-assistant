"""
chroma_client.py — Compatibility shim
======================================
هذا الملف موجود فقط للتوافق مع الكود القديم.
كل استدعاءات ChromaDB تمر الآن عبر core.chroma_manager.

لا تضع هنا أي Settings(...) أو PersistentClient(...) أبداً.
"""

from core.chroma_manager import get_chroma_client as get_client  # noqa: F401
