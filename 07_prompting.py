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
    """System prompt صارم جداً — يجيب فقط من البيانات المرفوعة ولا يستخدم معرفته العامة."""
    base = (
        "أنت مساعد قانوني متخصص في القوانين المصرية المتاحة في قاعدة المعرفة.\n"
        "مهمتك مساعدة المستخدمين في الأسئلة القانونية بأسلوب محادثة طبيعي وودّي.\n\n"
        "قواعد إجبارية — يجب اتباعها بدون استثناء:\n"
        "• ردّ على التحيات والأسئلة العامة (مثل مرحبا، كيف حالك، شكراً) بودّ وطلاقة.\n"
        "• اذكر رقم المادة صراحةً عند الاستشهاد بها (مثل: 'وفقًا للمادة 65').\n"
        "• اكتب بالعربية الفصحى الواضحة بأسلوب محادثي.\n"
        "• لا ترفض الإجابة فورًا — حاول فهم السؤال أولاً واطلب توضيحًا إذا لزم الأمر.\n"
        "• إذا كان السؤال غير قانوني وغير تحية، أخبر المستخدم بلطف أنك متخصص في القانون فقط.\n"
        "• إذا كان السؤال قانونيًا لكن لم يُعثر على سياق كافٍ، أخبر المستخدم بذلك واقترح رفع ملفات إضافية.\n"
        "• [حرج قصوى] لا تجب من معرفتك العامة تحت أي ظرف من الظروف.\n"
        "• [حرج قصوى] اجب فقط من النصوص المسترجعة من قاعدة المعرفة.\n"
        "• [حرج قصوى] إذا لم يكن هناك سياق قانوني مسترجع، لا تجب من معلوماتك الشخصية — قل إن المصدر غير مرفوع.\n"
    )
    if has_context:
        return base + (
            "\n--- السياق القانوني المسترجع --- (يجب أن تجيب فقط من هذا السياق ولا تستخدم أي معلومات أخرى):\n"
            "• اعتمد على النصوص الواردة في السياق فقط.\n"
            "• لا تستخدم أي معرفة خارجية أو تدريب مسبق.\n"
            "• اقتبس مباشرة من السياق وأشر لأرقام المواد.\n"
            "• إذا كان السياق غير كافٍ للإجابة، قل للمستخدم أن السياق غير كافٍ واطلب تفاصيل أكثر.\n"
            "• لا تضيف أي معلومات من معرفتك العامة."
        )
    return base + (
        "\nلم يُعثر على نصوص قانونية مباشرة لهذا الموضوع في قاعدة المعرفة الحالية.\n"
        "أخبر المستخدم أن المصدر القانوني المتعلق بسؤاله غير مرفوع بعد، واقترح عليه رفع الملف المناسب من الشريط الجانبي."
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
        user_message += f"\n\n--- السياق القانوني المسترجع --- (هذا هو المصدر الوحيد للإجابة):\n{context_text}\n--- نهاية السياق ---"
        user_message += (
            "\n\n[حرج قصوى] تعليمات إجبارية:"
            "\n1. اقرأ السياق القانوني أعلاه أولاً."
            "\n2. أجب فقط من المعلومات الواردة في السياق."
            "\n3. لا تستخدم أي معلومات أخرى أو معرفة عامة."
            "\n4. إذا كان السياق لا يحتوي على معلومات كافية للإجابة، قل: 'السياق غير كافٍ للإجابة على هذا السؤال.'"
            "\n5. لا تجب من معرفتك الشخصية أو تدريبك المسبق تحت أي ظرف."
        )
    else:
        user_message += "\n\n[حرج] لا توجد سياق قانوني متاح. لا تجب من معرفتك العامة."

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
