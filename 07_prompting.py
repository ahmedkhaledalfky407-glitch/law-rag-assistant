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
    """بناء system prompt صارم يقيّد الإجابة على المصادر المرفوعة فقط."""
    strict_rules = (
        "أنت مساعد قانوني متخصص، مهمتك الوحيدة هي الإجابة استناداً إلى النصوص القانونية "
        "الواردة في قاعدة المعرفة المقدمة إليك.\n\n"
        "القواعد الصارمة التي يجب الالتزام بها:\n"
        "1. لا تجب على أي سؤال خارج نطاق المصادر القانونية المرفوعة، مهما كان السؤال.\n"
        "2. لا تستخدم معرفتك العامة أو أي معلومات خارج السياق المقدم.\n"
        "3. إذا لم يكن السؤال متعلقاً بالمصادر القانونية المتاحة، رد بالضبط: "
        "'هذا السؤال خارج نطاق المصادر القانونية المتاحة. يمكنني فقط الإجابة على أسئلة تتعلق بالقوانين الموجودة في قاعدة المعرفة.'\n"
        "4. لا تناقش أي موضوع غير قانوني مهما طُلب منك.\n"
        "5. لا تقدم آراءً شخصية أو توصيات خارج النص القانوني.\n"
        "6. اكتب بالعربية الفصحى وأذكر رقم المادة صراحةً عند الاستشهاد بها.\n"
    )
    if has_context:
        return strict_rules + (
            "\nلديك سياق قانوني محدد — يجب أن تعتمد عليه حصراً في إجابتك.\n"
            "اقتبس من النص مباشرة وأشر إلى أرقام المواد الواردة فيه.\n"
            "إذا كان السياق غير كافٍ للإجابة الكاملة، قل ذلك صراحةً ولا تكمل من معرفتك."
        )
    return strict_rules + (
        "\nلم يُعثر على نصوص ذات صلة في قاعدة المعرفة لهذا السؤال.\n"
        "رد بالضبط: 'لا تتوفر في قاعدة المعرفة نصوص قانونية تتعلق بهذا السؤال. "
        "يرجى التأكد من رفع المصادر المناسبة أو إعادة صياغة سؤالك.'"
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
