"""واجهة Streamlit لمساعد قانوني ذكي يعتمد على RAG لقانون العمل المصري."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================
# تهيئة ChromaManager مرة واحدة عند أول import
# sys.modules يضمن أن هذا لن يُعاد تنفيذه في الـ reruns
# ============================================================
from core.chroma_manager import get_chroma_client as _init_chroma_singleton  # noqa: E402

_init_chroma_singleton()  # ينشئ الـ client مرة واحدة ويخزنه


def load_module(module_name: str, file_name: str):
    """تحميل وحدة Python من ملف باسم يبدأ برقم."""
    module_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"تعذر تحميل الوحدة: {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retrieve_context_module = load_module("retrieve_context", "06_retrieve_context.py")
generate_answer_module = load_module("prompting", "07_prompting.py")
documents_module = load_module("documents", "01_documents.py")
preprocessing_module = load_module("preprocessing", "02_preprocessing.py")
chunking_module = load_module("chunking", "03_chunking.py")
vector_module = load_module("vector_representation", "04_vector_representation.py")
chroma_module = load_module("create_chroma_store", "05_create_chroma_store.py")

retrieve_context = retrieve_context_module.retrieve_context
generate_answer = generate_answer_module.generate_answer
OPENROUTER_MODEL = generate_answer_module.OPENROUTER_MODEL

st.set_page_config(page_title="مساعد قانون العمل", page_icon="⚖️", layout="wide")


# ── Styles ──────────────────────────────────────────────────
def apply_rtl_style() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="st-"], .stApp {
            direction: rtl;
            text-align: right;
            font-family: 'Segoe UI', Tahoma, sans-serif;
        }
        .block-container { padding-top: 1.5rem; }
        .stFileUploader { direction: rtl; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Secrets helper ──────────────────────────────────────────
def get_secret_value(key: str, default: str = "") -> str:
    try:
        val = str(st.secrets.get(key, default))
        return val if val and val != "your_openrouter_api_key_here" else default
    except Exception:
        return default


# ── Session state ────────────────────────────────────────────
def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "upload_progress" not in st.session_state:
        st.session_state.upload_progress = None
    if "uploaded_sources" not in st.session_state:
        st.session_state.uploaded_sources = []
    if "openrouter_api_key" not in st.session_state:
        st.session_state.openrouter_api_key = (
            os.environ.get("OPENROUTER_API_KEY", "")
            or get_secret_value("OPENROUTER_API_KEY", "")
        )
    if "openrouter_model" not in st.session_state:
        st.session_state.openrouter_model = (
            os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
            or get_secret_value("OPENROUTER_MODEL", OPENROUTER_MODEL)
        )


# ── API key management ───────────────────────────────────────
def configure_api_keys(api_key: str | None = None, model: str | None = None) -> None:
    try:
        secret_api_key = get_secret_value("OPENROUTER_API_KEY", "")
        secret_model = get_secret_value("OPENROUTER_MODEL", OPENROUTER_MODEL)

        resolved_api_key = (
            api_key
            or st.session_state.get("openrouter_api_key", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
            or secret_api_key
        )
        resolved_model = (
            model
            or st.session_state.get("openrouter_model", "")
            or os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
            or secret_model
        )

        if api_key is not None:
            st.session_state.openrouter_api_key = api_key
        if model is not None:
            st.session_state.openrouter_model = model

        if resolved_api_key:
            os.environ["OPENROUTER_API_KEY"] = resolved_api_key
        if resolved_model:
            os.environ["OPENROUTER_MODEL"] = resolved_model

        generate_answer_module.configure_api_settings(
            api_key=resolved_api_key or None,
            model=resolved_model or None,
            secrets={
                "OPENROUTER_API_KEY": secret_api_key,
                "OPENROUTER_MODEL": secret_model,
            },
        )
    except Exception:
        pass


def _save_env_var(key: str, value: str) -> None:
    try:
        env_path = Path(__file__).with_name(".env")
        lines: list[str] = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                updated = True
                break
        if not updated:
            lines.append(f"{key}={value}")
        env_path.write_text("\n".join(lines), encoding="utf-8")
        os.environ[key] = value
        load_dotenv(str(env_path), override=True)
    except Exception:
        pass


# ── File upload ──────────────────────────────────────────────
def process_uploaded_file(uploaded_file) -> dict[str, Any]:
    stages = [
        ("حفظ الملف", "جاري حفظ الملف..."),
        ("تحميل الوثيقة", "جاري تحميل الوثيقة..."),
        ("تنظيف النص واستخراج المواد", "جاري استخراج المواد القانونية..."),
        ("تقسيم المواد إلى chunks", "جاري تقسيم المواد..."),
        ("إنشاء embeddings", "جاري إنشاء التمثيل المتجه..."),
        ("حفظ في قاعدة البيانات", "جاري حفظ في قاعدة البيانات..."),
    ]
    try:
        data_dir = Path(__file__).with_name("data")
        data_dir.mkdir(exist_ok=True)
        temp_path = data_dir / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())
        for stage_name, stage_msg in stages[:1]:
            st.session_state.upload_progress = (stage_name, stage_msg)

        documents = documents_module.load_documents(str(temp_path))
        if not documents or not documents[0].get("raw_text"):
            return {"success": False, "message": "تعذر قراءة الملف. تأكد من أنه ملف نصي صالح."}
        for stage_name, stage_msg in stages[1:2]:
            st.session_state.upload_progress = (stage_name, stage_msg)

        articles = preprocessing_module.process_documents(documents)
        if not articles:
            return {"success": False, "message": "لم يتم العثور على مواد قانونية في الملف."}
        for stage_name, stage_msg in stages[2:3]:
            st.session_state.upload_progress = (stage_name, stage_msg)

        chunks = chunking_module.chunk_articles(articles)
        if not chunks:
            return {"success": False, "message": "لم يتم إنشاء chunks من الملف."}
        for stage_name, stage_msg in stages[3:4]:
            st.session_state.upload_progress = (stage_name, stage_msg)

        embedded_chunks = vector_module.build_embeddings(chunks)
        for stage_name, stage_msg in stages[4:5]:
            st.session_state.upload_progress = (stage_name, stage_msg)

        result = chroma_module.create_or_update_chroma_store(
            embedded_chunks,
            persist_directory=os.environ.get("CHROMA_DB_PATH", "./chroma_db"),
            collection_name=os.environ.get("CHROMA_COLLECTION", "law_rag"),
            rebuild=False,
        )
        for stage_name, stage_msg in stages[5:]:
            st.session_state.upload_progress = (stage_name, stage_msg)

        return {
            "success": True,
            "message": f"✅ تمت إضافة '{uploaded_file.name}': {len(articles)} مادة، {len(chunks)} chunk",
            "articles_count": len(articles),
            "chunks_count": len(chunks),
        }
    except Exception as exc:
        return {"success": False, "message": f"حدث خطأ أثناء المعالجة: {exc}"}


# ── Connection status banner ─────────────────────────────────
def render_connection_status() -> None:
    api_key = st.session_state.get("openrouter_api_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
    model = st.session_state.get("openrouter_model", "") or os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)

    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        if api_key:
            st.success("🟢 النموذج متصل")
        else:
            st.error("🔴 النموذج غير متصل")
    with col2:
        st.caption(f"**النموذج:** `{model}`" if api_key else "أدخل مفتاح OpenRouter API من الشريط الجانبي")
    with col3:
        if st.button("🔌 اختبار الاتصال", use_container_width=True):
            if not api_key:
                st.warning("أدخل المفتاح أولاً من الشريط الجانبي")
            else:
                with st.spinner("جاري الاختبار..."):
                    try:
                        test_res = generate_answer_module.generate_answer("ping", [])
                        if test_res and "تعذر" not in test_res and "لم يتم" not in test_res:
                            st.success("✅ الاتصال يعمل")
                        else:
                            st.error(f"❌ {test_res}")
                    except Exception as exc:
                        st.error(f"❌ {exc}")
    st.divider()


# ── Sidebar ──────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.title("⚖️ مساعد قانون العمل")
        st.caption("مساعد ذكي يستند إلى نص قانون العمل المصري رقم 14 لسنة 2025")
        st.metric("عدد الأسئلة", st.session_state.question_count)
        st.divider()

        st.subheader("⚙️ إعدادات النموذج")
        api_key_input = st.text_input(
            "مفتاح OpenRouter API",
            value=st.session_state.get("openrouter_api_key", "") or get_secret_value("OPENROUTER_API_KEY", ""),
            type="password",
        )
        model_input = st.text_input(
            "اسم النموذج",
            value=st.session_state.get("openrouter_model", "") or get_secret_value("OPENROUTER_MODEL", OPENROUTER_MODEL),
        )

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("حفظ الإعدادات", use_container_width=True):
                if api_key_input:
                    _save_env_var("OPENROUTER_API_KEY", api_key_input)
                    _save_env_var("OPENROUTER_MODEL", model_input or OPENROUTER_MODEL)
                    configure_api_keys(api_key=api_key_input, model=model_input or OPENROUTER_MODEL)
                    st.success("تم الحفظ ✓")
                    st.rerun()
                else:
                    st.error("أدخل مفتاح API صالحًا.")
        with cols[1]:
            if st.button("اختبار", use_container_width=True):
                with st.spinner("جارٍ الاختبار..."):
                    try:
                        test_res = generate_answer_module.generate_answer("اختبار", [])
                        if test_res and "تعذر" not in test_res and "لم يتم" not in test_res:
                            st.success("✅ يعمل")
                        else:
                            st.error(f"❌ {test_res}")
                    except Exception as exc:
                        st.error(f"❌ {exc}")

        if st.button("مسح المحادثة", use_container_width=True):
            st.session_state.messages = []
            st.session_state.question_count = 0
            st.rerun()

        current_key = st.session_state.get("openrouter_api_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
        st.markdown(f"**الحالة:** {'✅ متصل' if current_key else '❌ غير متصل'}")
        if not current_key:
            st.warning("أضف OPENROUTER_API_KEY في البيئة أو الشريط الجانبي.")

        st.divider()
        st.subheader("📂 رفع مصادر جديدة")
        uploaded_file = st.file_uploader(
            "اختر ملف نصي",
            type=["txt"],
            help="ملف يحتوي على مواد قانونية بصيغة: المادة رقم X: النص",
        )
        if uploaded_file is not None:
            if st.button("إضافة المصدر", use_container_width=True, type="primary"):
                st.session_state.upload_progress = None
                stage_placeholder = st.empty()
                result = process_uploaded_file(uploaded_file)
                progress = st.session_state.get("upload_progress")
                if progress:
                    stage_placeholder.success(f"✅ {progress[0]}: {progress[1]}")
                if result["success"]:
                    st.session_state.uploaded_sources.append(uploaded_file.name)
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result["message"])

        if st.session_state.uploaded_sources:
            st.divider()
            st.subheader("📚 المصادر المضافة")
            for source in st.session_state.uploaded_sources:
                st.markdown(f"- ✅ {source}")


# ── Welcome ──────────────────────────────────────────────────
def render_welcome_message() -> None:
    if not st.session_state.messages:
        st.chat_message("assistant").write(
            "أهلاً بك. أنا مساعد متخصص في قانون العمل المصري لعام 2025. "
            "يمكنك سؤالي عن أي موضوع قانوني، وسأجيب بناءً على قاعدة المعرفة المتاحة. "
            "يمكنك أيضًا رفع ملفات نصية جديدة من الشريط الجانبي لإثراء قاعدة المعرفة."
        )


# ── Main ─────────────────────────────────────────────────────
def main() -> None:
    apply_rtl_style()
    initialize_session()
    configure_api_keys()
    render_connection_status()
    render_sidebar()
    render_welcome_message()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("اكتب سؤالك عن قانون العمل المصري..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.question_count += 1
        st.chat_message("user").write(prompt)

        context_chunks: list[dict[str, Any]] = []
        with st.spinner("جاري استرجاع السياق وإعداد الإجابة..."):
            try:
                context_chunks = retrieve_context(prompt, top_k=10)
                answer = generate_answer(prompt, context_chunks)
            except Exception as exc:
                answer = f"تعذر معالجة الطلب. التفاصيل: {exc}"

        st.chat_message("assistant").write(answer)
        if context_chunks:
            with st.expander("📚 المصادر المستخدمة", expanded=True):
                for item in context_chunks:
                    source = item.get("source", {})
                    st.write(
                        f"- المادة {source.get('article_number', 'غير محدد')} | "
                        f"الباب: {source.get('chapter', 'غير محدد')} | "
                        f"الكتاب: {source.get('book', 'غير محدد')} | "
                        f"التشابه: {item.get('similarity', 0):.2f}"
                    )
        else:
            st.caption("لا توجد مصادر. يمكنك رفع ملفات جديدة من الشريط الجانبي.")

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
