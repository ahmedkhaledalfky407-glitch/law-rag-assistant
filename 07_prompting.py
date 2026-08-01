"""إنشاء إجابات قانونية تعتمد على السياق المسترجَع فقط."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = "openai/gpt-4o-mini"


def configure_api_settings(api_key: str | None = None, model: str | None = None, secrets: dict[str, str] | None = None) -> tuple[str, str]:
    """تحديث إعدادات OpenRouter من البيئة أو Streamlit secrets أو القيم المقدمة."""
    global OPENROUTER_API_KEY, OPENROUTER_MODEL

    secret_store = secrets or {}

    if api_key is not None:
        OPENROUTER_API_KEY = api_key.strip()
    elif not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = (secret_store.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")).strip()

    if model is not None:
        OPENROUTER_MODEL = model.strip()
    elif not OPENROUTER_MODEL or OPENROUTER_MODEL == "openai/gpt-4o-mini":
        OPENROUTER_MODEL = (secret_store.get("OPENROUTER_MODEL") or os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)).strip()

    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENROUTER_MODEL"] = OPENROUTER_MODEL
    return OPENROUTER_API_KEY, OPENROUTER_MODEL


def get_api_config() -> tuple[str, str]:
    """إرجاع الإعدادات الحالية لواجهة OpenRouter."""
    return OPENROUTER_API_KEY, OPENROUTER_MODEL


configure_api_settings()


def build_system_prompt() -> str:
    """بناء system prompt عربي مرن يجمع بين السياق القانوني والمعرفة العامة."""
    return (
        "أنت مساعد قانوني عربي ودود متخصص في قانون العمل المصري. "
        "أجب بشكل واضح ومباشر، وركز أولًا على السياق القانوني المتاح، لكن لا تتردد في استخدام معرفتك العامة إذا كان السياق غير كافٍ. "
        "اكتب بالعربية البسيطة والفصحى مع شرح مناسب. "
        "إذا كانت هناك مادة قانونية مناسبة في السياق، اذكرها بشكل بسيط مثل: 'وفقًا للمادة 26' أو 'في النص القانوني'. "
        "إذا لم يكن السؤال واضحًا أو مبهمًا، اعترف بذلك بوضوح وقدم 2-3 أسئلة مقترحة تساعد على توضيح المطلوب. "
        "إذا لم يوجد سياق كافٍ، أجب بشكل عام وقل بوضوح أن الإجابة عامة وليست نصًا قانونيًا مباشرًا."
    )


def generate_answer(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """إنشاء إجابة نهائية باستخدام OpenRouter API أو إرجاع رسالة بديلة عند فشل الاتصال."""
    configure_api_settings()
    api_key, model_name = get_api_config()

    try:
        from openai import OpenAI
    except Exception as exc:
        return f"تعذر تهيئة العميل: {exc}"

    if not api_key:
        return "لم يتم توفير مفتاح OpenRouter. أضف المفتاح في البيئة أو في Streamlit secrets أو أدخله من الشريط الجانبي."

    context_text = "\n\n".join(
        [f"المادة {chunk.get('source', {}).get('article_number', 'غير محدد')}: {chunk.get('text', '')}" for chunk in context_chunks]
    )

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=model_name,
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
