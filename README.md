# ⚖️ مساعد قانون العمل المصري

مساعد ذكي يعتمد على تقنية RAG (Retrieval-Augmented Generation) للإجابة على الأسئلة المتعلقة بقانون العمل المصري رقم 14 لسنة 2025.

## ✨ المميزات

- **إجابات ذكية** تعتمد على قاعدة معرفة قانونية
- **رفع مصادر جديدة** - أضف ملفات نصية (txt) لإثراء قاعدة المعرفة
- **مرونة في الإجابات** - يجيب من السياق القانوني أو المعرفة العامة
- **واجهة عربية RTL** كاملة
- **سجل محادثة** مع إمكانية المسح

## 🚀 التشغيل المحلي

```bash
# 1. تثبيت المتطلبات
pip install -r requirements.txt

# 2. إنشاء ملف .env من القالب
cp .env.example .env
# ثم أضف مفتاح OpenRouter API

# 3. بناء قاعدة المعرفة
python build_index.py

# 4. تشغيل التطبيق
streamlit run streamlit_app.py
```

## ☁️ النشر على Streamlit Cloud

1. ارفع المشروع إلى GitHub
2. في [Streamlit Cloud](https://share.streamlit.io/)، أنشئ تطبيق جديد واربطه بالمستودع
3. أضف المتغيرات السرية في **Settings → Secrets**:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "openai/gpt-4o-mini"
EMBEDDING_PROVIDER = "local"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION = "law_rag"
```

## 📁 بنية المشروع

| الملف | الوظيفة |
|-------|---------|
| `01_documents.py` | تحميل الوثائق النصية |
| `02_preprocessing.py` | تنظيف النص واستخراج البنية |
| `03_chunking.py` | تقسيم المواد إلى chunks |
| `04_vector_representation.py` | إنشاء embeddings |
| `05_create_chroma_store.py` | إنشاء قاعدة المتجهات |
| `06_retrieve_context.py` | استرجاع السياق |
| `07_prompting.py` | توليد الإجابات |
| `streamlit_app.py` | واجهة التطبيق |
| `build_index.py` | بناء قاعدة المعرفة |

## 🔧 التقنيات المستخدمة

- **Streamlit** - واجهة المستخدم
- **ChromaDB** - قاعدة بيانات المتجهات
- **OpenRouter API** - نماذج الذكاء الاصطناعي
- **OpenAI SDK** - الاتصال بالنماذج