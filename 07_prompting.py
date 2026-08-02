"""إنشاء إجابات قانونية تعتمد على السياق المسترجَع فقط."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = "openai/gpt-4o-mini"


def configure_api_settings(api_key: str | None = None, model: str | None = None, secrets: dict[str, str] | None = None) -> tuple[str, str]:
    """تحديث إعدادات OpenRouter من البيئة أو Streamlit secrets أو القيم المقدمة."""
    global OPENROUTER_API_KEY, OPENROUTER_MODEL

    # إعادة تحميل .env في كل استدعاء لضمان التحديث الفوري
    load_dotenv(override=True)

    secret_store = secrets or {}

    # تجاهل قيمة placeholder
    def _clean(val: str) -> str:
        return val.strip() if val and val != "your_openrouter_api_key_here" else ""

    if api_key is not None:
        OPENROUTER_API_KEY = _clean(api_key)
    elif not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = _clean(
            secret_store.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
        )

    if model is not None:
        OPENROUTER_MODEL = model.strip()
    elif not OPENROUTER_MODEL or OPENROUTER_MODEL == "openai/gpt-4o-mini":
        OPENROUTER_MODEL = (
            secret_store.get("OPENROUTER_MODEL", "") or os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
        ).strip()

    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENROUTER_MODEL"] = OPENROUTER_MODEL
    return OPENROUTER_API_KEY, OPENROUTER_MODEL


def get_api_config() -> tuple[str, str]:
    """إرجاع الإعدادات الحالية لواجهة OpenRouter."""
    return OPENROUTER_API_KEY, OPENROUTER_MODEL


# تهيئة أولية عند تحميل الموديول
configure_api_settings()


def build_system_prompt(has_context: bool = False) -> str:
    """بناء system prompt عربي يختلف بحسب وجود سياق أو عدمه."""
    base = (
        "أنت مساعد قانوني عربي متخصص في القوانين المصرية.\n"
        "اكتب بالعربية الفصحى البسيطة وكن دقيقاً في النصوص القانونية.\n"
        "اذكر رقم المادة صراحةً عندما تستشهد بها، مثل: 'وفقًا للمادة 26'.\n"
    )
    if has_context:
        return base + (
            "لديك سياق قانوني محدد مسترجع من قاعدة المعرفة — يجب أن تعتمد عليه أساساً في إجابتك.\n"
            "اقتبس منه مباشرة وأشر إلى أرقام المواد الواردة فيه.\n"
            "لا تتجاهل السياق المقدم ولا تستبدله بمعلوماتك العامة إذا كان كافياً.\n"
            "إذا كان السياق جزئياً، أكمله من معرفتك مع الإشارة لذلك."
        )
    return base + (
        "لم يُعثر على نصوص قانونية محددة في قاعدة المعرفة لهذا السؤال.\n"
        "أجب بناءً على معرفتك العامة بالقانون المصري مع الإشارة إلى أن الإجابة عامة.\n"
        "إذا كان السؤال غير واضح، اطلب توضيحاً واقترح 2-3 أسئلة تساعد المستخدم."
    )


def generate_answer(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """إنشاء إجابة نهائية باستخدام OpenRouter API أو إرجاع رسالة بديلة عند فشل الاتصال."""
    # تحديث الإعدادات في كل استدعاء لضمان قراءة المفتاح الحديث
    api_key, model_name = configure_api_settings()

    try:
        from openai import OpenAI
    except Exception as exc:
        return f"تعذر تهيئة العميل: {exc}"

    if not api_key:
        return "لم يتم توفير مفتاح OpenRouter. أضف المفتاح في البيئة أو في Streamlit secrets أو أدخله من الشريط الجانبي."

    has_context = bool(context_chunks)
    context_text = "\n\n".join(
        [
            f"[مادة {chunk.get('source', {}).get('article_number', 'غير محدد')}]\n{chunk.get('text', '')}"
            for chunk in context_chunks
        ]
    )

    user_message = f"السؤال: {query}"
    if context_text:
        user_message += f"\n\n--- السياق القانوني المسترجع ---\n{context_text}\n--- نهاية السياق ---"

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": build_system_prompt(has_context=has_context)},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "لم يتم إنشاء إجابة."
    except Exception as exc:
        return f"تعذر الاتصال بالنموذج: {exc}"


if __name__ == "__main__":
    print(generate_answer("ما حكم العمل؟", [{"text": "المادة الأولى: تبدأ أحكام هذا القانون من تاريخ العمل.", "source": {"article_number": "1"}}]))
