"""إنشاء إجابات قانونية تعتمد على السياق المسترجَع فقط."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = "openai/gpt-4o"


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
    """System prompt طبيعي ومتخصص في قانون العمل المصري."""
    base = (
        "أنت مساعد قانوني ذكي ومتخصص في قانون العمل المصري رقم 14 لسنة 2025 والمصادر القانونية المرفوعة.\n"
        "مهمتك الإجابة على الأسئلة المتعلقة بقانون العمل المصري فقط، لكنك أيضاً مساعد ودود في الحوار العام.\n\n"
        "طريقة العمل:\n"
        "• كن ودوداً وطبيعياً في الحوار.\n"
        "• اجب من النصوص المسترجعة من قاعدة المعرفة، واذكر رقم المادة عند الاستشهاد.\n"
        "• إذا لم يكن هناك سياق قانوني، والسؤال تحية أو سؤال عام عن النظام، رد ببساطة وودية.\n"
        "• إذا لم يكن هناك سياق، والسؤال قانوني فعلاً، قل: 'لا توجد مادة قانونية مطابقة داخل قاعدة المعرفة الحالية.'\n"
        "• لا تخترع مواد قانونية أو أرقام مواد أو عقوبات.\n"
        "• اكتب بالعربية الفصحى الواضحة بأسلوب بسيط ومفهوم.\n"
    )
    if has_context:
        return base + (
            "\nالسياق القانوني متاح — اجب منه مباشرة. "
            "إذا كان السياق غير كافٍ، قل: 'المواد المسترجعة لا تحتوي على إجابة كاملة لهذا السؤال.'\n"
        )
    return base + (
        "\nلا توجد مواد قانونية متاحة حالياً. "
        "إذا كان السؤال تحية أو سؤال عام، رد بودية وطبيعية. "
        "إذا كان السؤال قانونياً، رد فقط: 'لا توجد مادة قانونية مطابقة داخل قاعدة المعرفة الحالية.'\n"
    )


def generate_answer(query: str, context_chunks: list[dict[str, Any]], history: list[dict[str, str]] | None = None) -> str:
    """إنشاء إجابة نهائية باستخدام OpenRouter API مع تاريخ المحادثة."""
    api_key, model_name = configure_api_settings()
    logger.info("generate_answer: query='%s', has_context=%d, model=%s", query[:50], len(context_chunks), model_name)

    try:
        from openai import OpenAI
    except Exception as exc:
        logger.error("generate_answer: failed to import OpenAI - %s", exc)
        return f"تعذر تهيئة العميل: {exc}"

    if not api_key:
        logger.warning("generate_answer: no API key configured")
        return "لم يتم توفير مفتاح OpenRouter. أضف المفتاح في البيئة أو في Streamlit secrets أو أدخله من الشريط الجانبي."

    has_context = bool(context_chunks)
    context_text = "\n\n".join(
        [
            f"[مادة {chunk.get('source', {}).get('article_number', 'غير محدد')} | المصدر: {chunk.get('source', {}).get('source_file', 'غير محدد')}]\n{chunk.get('text', '')}"
            for chunk in context_chunks
        ]
    )

    system_prompt = build_system_prompt(has_context=has_context)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history[-10:])

    user_message = f"السؤال: {query}\n\n"
    if context_text:
        user_message += (
            f"المصادر القانونية المتاحة:\n"
            f"{context_text}\n\n"
            f"أجب على السؤال مباشرة من المصادر أعلاه. "
            f"إذا لم تجد إجابة كافية، قل ببساطة: 'المواد المسترجعة لا تحتوي على إجابة كاملة.' "
            f"لا تخترع معلومات.\n"
        )
    else:
        user_message += (
            "لا توجد مواد قانونية متاحة حالياً. "
            "إذا كان السؤال تحية أو سؤال عام عن النظام، رد بودية وطبيعية. "
            "إذا كان السؤال قانونياً، رد فقط: 'لا توجد مادة قانونية مطابقة داخل قاعدة المعرفة الحالية.'\n"
        )

    messages.append({"role": "user", "content": user_message})

    logger.info("generate_answer: sending to model with %d chars of context, %d history messages", len(context_text), len(history or []))

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5,
            max_tokens=1200,
        )
        answer = response.choices[0].message.content or "لم يتم إنشاء إجابة."
        logger.info("generate_answer: received answer (%d chars)", len(answer))
        return answer
    except Exception as exc:
        logger.error("generate_answer: API call failed - %s", exc)
        return f"تعذر الاتصال بالنموذج: {exc}"


if __name__ == "__main__":
    print(generate_answer("ما حكم العمل؟", [{"text": "المادة الأولى: تبدأ أحكام هذا القانون من تاريخ العمل.", "source": {"article_number": "1"}}]))
