"""إنشاء إجابات قانونية تعتمد على السياق المسترجَع فقط."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def build_system_prompt() -> str:
    """بناء system prompt عربي مرن يجمع بين السياق القانوني والمعرفة العامة."""
    return (
        "أنت مساعد قانوني متخصص في قانون العمل المصري. "
        "أجب بالاعتماد على السياق القانوني المسترجَع أولًا، وإذا لم يكن كافيًا يمكنك استخدام معرفتك العامة للمساعدة. "
        "اكتب بالعربية الفصحى وبشكل واضح ومفهوم. "
        "في كل إجابة اذكر رقم المادة القانونية المستخدمة صراحةً مثل: 'وفقًا للمادة 26' إذا كانت متوفرة في السياق. "
        "إذا لم يكن هناك سياق كافٍ، أجب من معرفتك العامة مع توضيح أن الإجابة عامة وليست من النص القانوني المسترجَع. "
        "كن مرنًا في فهم الأسئلة، وإذا كان السؤال غير واضح اطلب توضيحًا أو أعد صياغته بطريقة مفهومة."
    )


def generate_answer(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """إنشاء إجابة نهائية باستخدام OpenRouter API أو إرجاع رسالة بديلة عند فشل الاتصال."""
    try:
        from openai import OpenAI
    except Exception as exc:
        return f"تعذر تهيئة العميل: {exc}"

    if not OPENROUTER_API_KEY:
        return "لم يتم توفير مفتاح OpenRouter. أضف OPENROUTER_API_KEY في البيئة أو في Streamlit secrets."

    context_text = "\n\n".join(
        [f"المادة {chunk.get('source', {}).get('article_number', 'غير محدد')}: {chunk.get('text', '')}" for chunk in context_chunks]
    )

    try:
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
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
