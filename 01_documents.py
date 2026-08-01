"""تحميل الوثائق النصية لقانون العمل المصري بصياغة عربية واضحة."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def load_documents(path: str) -> list[dict[str, Any]]:
    """اقرأ ملفًا نصيًا عربيًا وأعده على شكل قائمة من الوثائق ذات metadata أولية."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"الملف غير موجود: {path}")

        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        return [
            {
                "source_file": file_path.name,
                "raw_text": text,
                "line_count": len(lines),
                "path": str(file_path),
            }
        ]
    except Exception as exc:
        return [{"source_file": Path(path).name, "raw_text": "", "line_count": 0, "path": path, "error": str(exc)}]


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "law_sample.txt"
    if not os.path.exists(target):
        Path(target).write_text(
            "الكتاب الأول\nالباب الأول\nالمادة رقم 1: يبدأ العمل من تاريخ التعيين.\nالمادة الثانية: تلتزم الجهات بأحكام هذا القانون.\n",
            encoding="utf-8",
        )
    documents = load_documents(target)
    print(f"تم تحميل {len(documents)} وثيقة/وثائق من {target}")
    for doc in documents:
        print(f"- الملف: {doc['source_file']} | الأسطر: {doc['line_count']}")
