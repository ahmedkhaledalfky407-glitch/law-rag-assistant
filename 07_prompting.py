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
    """System prompt صارم جداً — يجيب فقط من المصادر المسترجعة."""
    base = (
        "أنت مساعد قانوني متخصص حصرياً في قانون العمل المصري رقم 14 لسنة 2025 والمصادر القانونية المرفوعة في قاعدة المعرفة.\n"
        "مهمتك الإجابة على الأسئلة المتعلقة بقانون العمل المصري فقط، وب ONLY من النصوص المسترجعة من قاعدة المعرفة.\n\n"
        "قواعد إجبارية — يجب اتباعها بدون استثناء:\n"
        "• لا تجب من معرفتك العامة تحت أي ظرف من الظروف.\n"
        "• لا تجب من تدريبك المسبق أو معلوماتك الشخصية تحت أي ظرف.\n"
        "• اجب فقط من النصوص المسترجعة من قاعدة المعرفة.\n"
        "• إذا لم يكن هناك سياق قانوني مسترجع، لا تجب — قل إن المصدر غير مرفوع.\n"
        "• إذا كان السياق المسترجع لا يحتوي على إجابة كافية، قل ذلك بوضوح.\n"
        "• لا تخترع مواد قانونية أو أرقام مواد أو عقوبات أو تواريخ أو استثناءات.\n"
        "• لا تستخدم أي قانون خارج المصادر المسترجعة.\n"
        "• إذا كان السؤال يتعلق بأي قانون آخر غير قانون العمل ولم يوجد في المصادر، قل: 'هذا النظام متخصص فقط في قانون العمل المصري رقم 14 لسنة 2025 والمصادر القانونية المرفوعة.'\n"
        "• اذكر رقم المادة صراحةً عند الاستشهاد بها.\n"
        "• اكتب بالعربية الفصحى الواضحة.\n"
    )
    if has_context:
        return base + (
            "\n--- تنسيق الإجابة الإجباري ---\n"
            "يجب أن تتبع هذا التنسيق بالضبط:\n\n"
            "1. الإجابة المختصرة\n\n"
            "2. الأساس القانوني\n"
            "- المادة رقم (...)\n"
            "- النص المستند إليه\n\n"
            "3. الشرح القانوني\n"
            "اشرح فقط ما يوجد في المواد المسترجعة.\n\n"
            "4. الاستثناءات\n"
            "اذكر فقط إذا وجدت صراحة في النصوص.\n\n"
            "5. الإجراءات\n"
            "اذكر فقط إذا وجدت صراحة في النصوص.\n\n"
            "6. المصادر\n"
            "• اسم الوثيقة\n"
            "• رقم المادة\n"
            "• رقم الصفحة (إذا توفر)\n"
            "• معرف الـ chunk\n\n"
            "--- قواعد صارمة ---\n"
            "• لا تخلط بين مواد قانونية مختلفة.\n"
            "• إذا بدت المواد متناقضة، قل: 'توجد أكثر من مادة قانونية ذات صلة ويجب الرجوع للنصوص كاملة.'\n"
            "• كل بيان قانوني يجب أن يشير إلى المادة والمصدر.\n"
            "• بدون استشهاد = لا تذكر البيان.\n"
            "• لا تختم بإجابة خارج السياق.\n"
        )
    return base + (
        "\nلا توجد مادة قانونية مطابقة داخل قاعدة المعرفة الحالية.\n"
        "لا تحاول الإجابة. رد بهذه الجملة فقط: 'لا توجد مادة قانونية مطابقة داخل قاعدة المعرفة الحالية.'\n"
    )


def generate_answer(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """إنشاء إجابة نهائية باستخدام OpenRouter API أو إرجاع رسالة بديلة عند فشل الاتصال."""
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

    if not context_chunks:
        logger.warning("generate_answer: empty context")
        return "لا توجد مادة قانونية مطابقة داخل قاعدة المعرفة الحالية."

    has_context = bool(context_chunks)
    context_text = "\n\n".join(
        [
            f"[مادة {chunk.get('source', {}).get('article_number', 'غير محدد')} | المصدر: {chunk.get('source', {}).get('source_file', 'غير محدد')} | معرف: {chunk.get('source', {}).get('chunk_id', 'غير محدد')}]\n{chunk.get('text', '')}"
            for chunk in context_chunks
        ]
    )

    user_message = f"السؤال: {query}\n\n"
    user_message += (
        f"--- السياق القانوني المسترجع --- (هذا هو المصدر الوحيد للإجابة — ممنوع استخدام أي معلومة خارجية):\n"
        f"{context_text}\n"
        f"--- نهاية السياق ---\n\n"
        f"[حرج قصوى] تعليمات إجبارية:\n"
        f"1. اقرأ السياق القانوني أعلاه بالكامل أولاً.\n"
        f"2. حدد أي جزء من السياق يتعلق بالسؤال.\n"
        f"3. اجب فقط من المعلومات الموجودة في السياق.\n"
        f"4. استخدم التنسيق الإجباري التالي:\n"
        f"   1. الإجابة المختصرة\n"
        f"   2. الأساس القانوني\n"
        f"      - المادة رقم (...)\n"
        f"      - النص المستند إليه\n"
        f"   3. الشرح القانوني\n"
        f"   4. الاستثناءات (إذا وجدت)\n"
        f"   5. الإجراءات (إذا وجدت)\n"
        f"   6. المصادر\n"
        f"      • اسم الوثيقة\n"
        f"      • رقم المادة\n"
        f"      • رقم الصفحة (إذا توفر)\n"
        f"      • معرف الـ chunk\n"
        f"5. إذا لم يجد السياق معلومات كافية، رد: 'المواد القانونية المسترجعة لا تحتوي على إجابة كاملة لهذا السؤال.'\n"
        f"6. لا تجب من معرفتك الشخصية أو تدريبك المسبق تحت أي ظرف.\n"
        f"7. لا تضيف أي معلومات من خارج السياق.\n"
        f"8. لا تخترع مواد أو أرقام مواد أو عقوبات أو تواريخ.\n"
        f"9. إذا ظهرت مواد متناقضة، قل: 'توجد أكثر من مادة قانونية ذات صلة ويجب الرجوع للنصوص كاملة.'\n"
    )

    logger.info("generate_answer: sending to model with %d chars of context", len(context_text))

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": build_system_prompt(has_context=has_context)},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )
        answer = response.choices[0].message.content or "لم يتم إنشاء إجابة."
        logger.info("generate_answer: received answer (%d chars)", len(answer))
        return answer
    except Exception as exc:
        logger.error("generate_answer: API call failed - %s", exc)
        return f"تعذر الاتصال بالنموذج: {exc}"


if __name__ == "__main__":
    print(generate_answer("ما حكم العمل؟", [{"text": "المادة الأولى: تبدأ أحكام هذا القانون من تاريخ العمل.", "source": {"article_number": "1"}}]))
