"""إنشاء إجابات قانونية تعتمد على السياق المسترجَع فقط."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# قراءة المفتاح من البيئة أو من secrets.toml
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# محاولة قراءة من secrets.toml إذا كان الملف موجودًا
_secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
if not OPENROUTER_API_KEY and _secrets_path.exists():
    try:
        import tomllib
        with open(_secrets_path, "rb") as f:
            _secrets = tomllib.load(f)
        _key = _secrets.get("OPENROUTER_API_KEY", "")
        # تجاهل القيمة الافتراضية placeholder
        if _key and _key != "your_openrouter_api_key_here":
            OPENROUTER_API_KEY = _key
            OPENROUTER_MODEL = _secrets.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
    except Exception:
        pass


def build_system_prompt() -> str:
    """بناء system prompt عربي مرن يجمع بين السياق القانوني والمعرفة العامة."""
    return (
        "أنت مساعد قانوني متخصص في قانون العمل المصري. "
        "اعتمد أولًا على السياق القانوني المسترجَع. إذا لم يكن كافياً، يمكنك استخدام معرفتك العامة مع وسم أن الإجابة عامة. "
        "اكتب بالعربية الفصحى وبشكل واضح ومفهوم. "
        "اذكر رقم المادة القانونية المستخدمة صراحةً عندما تتوفر في السياق، مثل: 'وفقًا للمادة 26'. "
        "إذا كان السؤال غير واضح، أعد صياغته باختصار ثم اقترح 2-3 أسئلة توضيحية للمستخدم. "
        "إذا لم تكفِ المعلومات لإعطاء حكم قاطع فقدم إجابة استرشادية قصيرة وأوضح حدود اليقين، ثم اطلب مزيداً من التفاصيل."
    )


def generate_answer(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """إنشاء إجابة نهائية باستخدام OpenRouter API أو إرجاع رسالة بديلة عند فشل الاتصال."""
    try:
        from openai import OpenAI
    except Exception as exc:
        return f"تعذر تهيئة العميل: {exc}"

    # إعادة تحميل .env في كل استدعاء لضمان التحديث الفوري
    load_dotenv(override=True)
    api_key = os.environ.get("OPENROUTER_API_KEY", "") or OPENROUTER_API_KEY
    model = os.environ.get("OPENROUTER_MODEL", "") or OPENROUTER_MODEL

    if not api_key:
        return "لم يتم توفير مفتاح OpenRouter. أضف OPENROUTER_API_KEY في البيئة أو في Streamlit secrets."

    context_text = "\n\n".join(
        [f"المادة {chunk.get('source', {}).get('article_number', 'غير محدد')}: {chunk.get('text', '')}" for chunk in context_chunks]
    )

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": f"السؤال: {query}\n\nالسياق:\n{context_text}"},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "لم يتم إنشاء إجابة." 
    except Exception as exc:
        return f"تعذر الاتصال بالنموذج: {exc}"


if __name__ == "__main__":
    print(generate_answer("ما حكم العمل؟", [{"text": "المادة الأولى: تبدأ أحكام هذا القانون من تاريخ العمل.", "source": {"article_number": "1"}}]))
