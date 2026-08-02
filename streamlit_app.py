"""واجهة Streamlit لمساعد قانوني ذكي يعتمد على RAG لقانون العمل المصري."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv


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


def apply_rtl_style() -> None:
    """تطبيق CSS بسيط لدعم الاتجاه العربي في الواجهة."""
    st.markdown(
        """
        <style>
        html, body, [class*="st-"], .stApp {
            direction: rtl;
            text-align: right;
            font-family: 'Segoe UI', Tahoma, sans-serif;
        }
        .block-container {
            padding-top: 1.5rem;
        }
        .stFileUploader {
            direction: rtl;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session() -> None:
    """تهيئة حالة الجلسة للحفاظ على سجل المحادثة."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "uploaded_sources" not in st.session_state:
        st.session_state.uploaded_sources = []


def configure_api_keys() -> None:
    """قراءة المفتاح من البيئة أو Streamlit secrets مع fallback واضح."""
    try:
        if not os.environ.get("OPENROUTER_API_KEY", ""):
            _key = st.secrets.get("OPENROUTER_API_KEY", "")
            # تجاهل القيمة الافتراضية placeholder
            if _key and _key != "your_openrouter_api_key_here":
                os.environ["OPENROUTER_API_KEY"] = _key
        if not os.environ.get("OPENROUTER_MODEL", ""):
            os.environ["OPENROUTER_MODEL"] = st.secrets.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
    except Exception:
        pass


def _save_env_var(key: str, value: str) -> None:
    """Save or update a key in the local .env file and set it in the current process env."""
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
        # Best-effort; do not crash the app on save failure
        pass


def process_uploaded_file(uploaded_file) -> dict[str, Any]:
    """معالجة ملف مرفوع وإضافته إلى قاعدة المعرفة."""
    try:
        # حفظ الملف مؤقتًا
        temp_path = Path("data") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        temp_path.write_bytes(uploaded_file.getvalue())

        # 1. تحميل المستند
        documents = documents_module.load_documents(str(temp_path))
        if not documents or not documents[0].get("raw_text"):
            return {"success": False, "message": "تعذر قراءة الملف. تأكد من أنه ملف نصي صالح."}

        # 2. المعالجة والتنظيف
        articles = preprocessing_module.process_documents(documents)
        if not articles:
            return {"success": False, "message": "لم يتم العثور على مواد قانونية في الملف."}

        # 3. تقسيم إلى chunks
        chunks = chunking_module.chunk_articles(articles)
        if not chunks:
            return {"success": False, "message": "لم يتم إنشاء chunks من الملف."}

        # 4. إنشاء embeddings
        embedded_chunks = vector_module.build_embeddings(chunks)

        # 5. إضافة إلى ChromaDB
        result = chroma_module.create_or_update_chroma_store(
            embedded_chunks,
            persist_directory=os.environ.get("CHROMA_DB_PATH", "./chroma_db"),
            collection_name=os.environ.get("CHROMA_COLLECTION", "law_rag"),
        )

        return {
            "success": True,
            "message": f"تمت إضافة الملف بنجاح: {len(articles)} مادة، {len(chunks)} chunk",
            "articles_count": len(articles),
            "chunks_count": len(chunks),
        }
    except Exception as exc:
        return {"success": False, "message": f"حدث خطأ أثناء المعالجة: {exc}"}


def render_sidebar() -> None:
    """عرض الشريط الجانبي مع معلومات المشروع وأزرار التحكم."""
    with st.sidebar:
        st.title("⚖️ مساعد قانون العمل")
        st.caption("مساعد ذكي يستند إلى نص قانون العمل المصري رقم 14 لسنة 2025")
        st.metric("عدد الأسئلة", st.session_state.question_count)
        st.divider()
        st.subheader("⚙️ إعدادات النموذج")
        api_key_input = st.text_input("مفتاح OpenRouter API", value=os.environ.get("OPENROUTER_API_KEY", ""), type="password")
        model_input = st.text_input("اسم النموذج", value=os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL))
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("حفظ مفتاح API", use_container_width=True):
                if api_key_input:
                    _save_env_var("OPENROUTER_API_KEY", api_key_input)
                    _save_env_var("OPENROUTER_MODEL", model_input or OPENROUTER_MODEL)
                    st.success("تم حفظ مفتاح API وإعدادات النموذج في .env")
                    st.experimental_rerun()
                else:
                    st.error("أدخل مفتاح API صالحًا قبل الحفظ.")
        with cols[1]:
            if st.button("اختبار الاتصال", use_container_width=True):
                with st.spinner("جارٍ اختبار الاتصال بالنموذج..."):
                    try:
                        test_res = generate_answer_module.generate_answer("اختبار اتصال", [])
                        if test_res and not test_res.startswith("تعذر") and "لم يتم توفير مفتاح" not in test_res:
                            st.success("نجح الاتصال: النموذج يستجيب")
                        else:
                            st.error(f"فشل الاتصال: {test_res}")
                    except Exception as exc:
                        st.error(f"فشل اختبار الاتصال: {exc}")

        if st.button("مسح المحادثة", use_container_width=True):
            st.session_state.messages = []
            st.session_state.question_count = 0
            st.rerun()

        api_status = "متصل" if os.environ.get("OPENROUTER_API_KEY", "") else "غير متصل"
        st.markdown(f"**حالة النموذج:** {api_status}")
        if not os.environ.get("OPENROUTER_API_KEY", ""):
            st.warning("لم يتم العثور على OPENROUTER_API_KEY. أضف المفتاح في البيئة أو في Streamlit secrets قبل استخدام النموذج.")

        st.divider()
        st.subheader("📂 رفع مصادر جديدة")
        st.caption("ارفع ملفات نصية (txt) لإضافتها إلى قاعدة المعرفة")
        uploaded_file = st.file_uploader(
            "اختر ملف نصي",
            type=["txt"],
            help="ارفع ملف نصي يحتوي على مواد قانونية بصيغة: المادة رقم X: النص",
        )
        if uploaded_file is not None:
            if st.button("إضافة المصدر", use_container_width=True, type="primary"):
                with st.spinner("جاري معالجة الملف وإضافته إلى قاعدة المعرفة..."):
                    result = process_uploaded_file(uploaded_file)
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


def render_welcome_message() -> None:
    """عرض رسالة ترحيبية عند أول تشغيل."""
    if not st.session_state.messages:
        st.chat_message("assistant").write(
            "أهلاً بك. أنا مساعد متخصص في قانون العمل المصري لعام 2025. "
            "يمكنك سؤالي عن أي موضوع قانوني، وسأجيب بناءً على قاعدة المعرفة المتاحة. "
            "يمكنك أيضًا رفع ملفات نصية جديدة من الشريط الجانبي لإثراء قاعدة المعرفة."
        )


def main() -> None:
    """المنطق الرئيسي للواجهة."""
    apply_rtl_style()
    initialize_session()
    configure_api_keys()
    render_sidebar()
    render_welcome_message()

    if not os.environ.get("OPENROUTER_API_KEY", ""):
        st.info("الواجهة جاهزة، لكن الإرسال إلى النموذج يتطلب مفتاح OpenRouter. سيتم عرض رسالة بديلة إذا لم يتوفر المفتاح.")

    if prompt := st.chat_input("اكتب سؤالك عن قانون العمل المصري..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.question_count += 1
        st.chat_message("user").write(prompt)

        context_chunks: list = []
        with st.spinner("جاري استرجاع السياق وإعداد الإجابة..."):
            try:
                context_chunks = retrieve_context(prompt, top_k=5)
                answer = generate_answer(prompt, context_chunks)
            except Exception as exc:
                answer = f"تعذر معالجة الطلب. يرجى المحاولة لاحقًا. التفاصيل: {exc}"

        st.chat_message("assistant").write(answer)
        with st.expander("المصادر المستخدمة", expanded=True):
            if context_chunks:
                for item in context_chunks:
                    source = item.get("source", {})
                    st.write(
                        f"- المادة {source.get('article_number', 'غير محدد')} | الباب: {source.get('chapter', 'غير محدد')} | الكتاب: {source.get('book', 'غير محدد')} | التشابه: {item.get('similarity', 0):.2f}"
                    )
            else:
                st.write("لا توجد مصادر متاحة حاليًا. يمكنك رفع ملفات جديدة من الشريط الجانبي.")

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()