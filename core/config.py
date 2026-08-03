"""
وحدة إدارة الإعدادات والمفاتيح الموحدة (Configuration Manager)
=====================================================
تجمع وتقرأ الإعدادات بحسب الأولوية التالية:
1. st.session_state (إذا توفرت مدخلات الواجهة)
2. os.environ (متغيرات البيئة)
3. st.secrets (إعدادات Streamlit Cloud Secrets)
4. ملف .env (البيئة المحلية)

وتضمن تحديث os.environ لتستفيد منها جميع المكتبات مثل OpenAI SDK و ChromaDB.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# تحميل ملف .env إن وجد دون مسح القيم الحالية
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH, override=False)
    except Exception as exc:
        logger.warning("CoreConfig: تعذر تحميل .env - %s", exc)


def _clean_val(val: Any, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    if not s or s == "your_openrouter_api_key_here":
        return default
    return s


def _get_from_streamlit_secrets(key: str) -> str:
    """استخراج المفتاح من Streamlit secrets بأشكال هيكلية متعددة."""
    try:
        import streamlit as st
        # 1. مفتاح مباشر بنفس الاسم
        if key in st.secrets:
            v = _clean_val(st.secrets.get(key))
            if v:
                return v
        # 2. هيكل مفاتيح فرعية مثل [openrouter] api_key
        sub_map = {
            "OPENROUTER_API_KEY": [("openrouter", "api_key"), ("openrouter", "key"), ("OPENROUTER", "API_KEY")],
            "OPENROUTER_MODEL": [("openrouter", "model"), ("OPENROUTER", "MODEL")],
            "EMBEDDING_PROVIDER": [("embedding", "provider"), ("EMBEDDING", "PROVIDER")],
            "EMBEDDING_MODEL": [("embedding", "model"), ("EMBEDDING", "MODEL")],
            "CHROMA_DB_PATH": [("chroma", "path"), ("chroma", "db_path")],
            "CHROMA_COLLECTION": [("chroma", "collection")],
        }
        if key in sub_map:
            for sec, sub_k in sub_map[key]:
                if sec in st.secrets:
                    section = st.secrets[sec]
                    if isinstance(section, dict) or hasattr(section, "get"):
                        v = _clean_val(section.get(sub_k))
                        if v:
                            return v
    except Exception:
        pass
    return ""


def get_config_var(key: str, default: str = "") -> str:
    """استرجاع قيمة متغير بإتباع ترتيب الأولويات الموحد."""
    # 1. Check Streamlit session_state
    try:
        import streamlit as st
        session_key = key.lower()
        if session_key in st.session_state and st.session_state[session_key]:
            v = _clean_val(st.session_state[session_key])
            if v:
                return v
    except Exception:
        pass

    # 2. Check os.environ
    env_v = _clean_val(os.environ.get(key, ""))
    if env_v:
        return env_v

    # 3. Check Streamlit secrets
    secret_v = _get_from_streamlit_secrets(key)
    if secret_v:
        return secret_v

    return default


def get_openrouter_api_key() -> str:
    return get_config_var("OPENROUTER_API_KEY", "")


def get_openrouter_model() -> str:
    return get_config_var("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def get_embedding_provider() -> str:
    return get_config_var("EMBEDDING_PROVIDER", "openai").lower()


def get_embedding_model() -> str:
    return get_config_var("EMBEDDING_MODEL", "openai/text-embedding-3-small")


def sync_config() -> dict[str, str]:
    """مزامنة كل الإعدادات في os.environ وإعادتها كقاموس."""
    api_key = get_openrouter_api_key()
    model = get_openrouter_model()
    emb_provider = get_embedding_provider()
    emb_model = get_embedding_model()
    chroma_path = get_config_var("CHROMA_DB_PATH", "./chroma_db")
    chroma_collection = get_config_var("CHROMA_COLLECTION", "law_rag")

    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    if model:
        os.environ["OPENROUTER_MODEL"] = model
    os.environ["EMBEDDING_PROVIDER"] = emb_provider
    os.environ["EMBEDDING_MODEL"] = emb_model
    os.environ["CHROMA_DB_PATH"] = chroma_path
    os.environ["CHROMA_COLLECTION"] = chroma_collection

    return {
        "OPENROUTER_API_KEY": api_key,
        "OPENROUTER_MODEL": model,
        "EMBEDDING_PROVIDER": emb_provider,
        "EMBEDDING_MODEL": emb_model,
        "CHROMA_DB_PATH": chroma_path,
        "CHROMA_COLLECTION": chroma_collection,
    }
