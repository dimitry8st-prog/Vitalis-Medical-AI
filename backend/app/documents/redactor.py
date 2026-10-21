from __future__ import annotations

import re
from collections import Counter

from app.documents.models import DocumentPage, PiiReport

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:\+7|8)[\s()\-]*\d{3}[\s()\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)"
        ),
    ),
    ("snils", re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}[ -]\d{2}(?!\d)")),
    (
        "passport",
        re.compile(
            r"(?i)(?:(?:паспорт|серия\s+и\s+номер)\s*[:№]?\s*)\d{2}\s?\d{2}\s?\d{6}"
        ),
    ),
    (
        "oms",
        re.compile(r"(?i)(?:(?:полис\s+)?ОМС\s*[:№]?\s*)\d{16}"),
    ),
    (
        "birth_date",
        re.compile(
            r"(?i)(?:(?:дата\s+рождения|д\.?\s*р\.?)\s*[:]?\s*)\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
        ),
    ),
    (
        "patient_id",
        re.compile(
            r"(?i)(?:(?:№\s*)?(?:карты|истории\s+болезни|медкарты)\s*[:№]?\s*)[A-ZА-Я0-9\-/]{4,}"
        ),
    ),
    (
        "name",
        re.compile(
            r"(?im)^\s*(?:Ф\.?И\.?О\.?|пациент(?:ка)?|больной|больная)\s*:\s*[^\n]{3,120}$"
        ),
    ),
    (
        "address",
        re.compile(
            r"(?im)^\s*(?:адрес(?:\s+регистрации|\s+проживания)?)\s*:\s*[^\n]{5,200}$"
        ),
    ),
    (
        "full_name",
        re.compile(
            r"\b[А-ЯЁ][а-яё-]{1,40}\s+[А-ЯЁ][а-яё-]{1,40}\s+[А-ЯЁ][а-яё-]{2,40}(?:ович|евич|ич|овна|евна|ична)\b"
        ),
    ),
)


def redact_pages(
    pages: tuple[DocumentPage, ...],
) -> tuple[tuple[DocumentPage, ...], PiiReport]:
    counts: Counter[str] = Counter()
    redacted_pages: list[DocumentPage] = []

    for page in pages:
        text = page.text
        for kind, pattern in _PATTERNS:
            replacement = f"[REDACTED_{kind.upper()}]"

            def replace(
                _match: re.Match[str],
                *,
                _kind: str = kind,
                _replacement: str = replacement,
            ) -> str:
                counts[_kind] += 1
                return _replacement

            text = pattern.sub(replace, text)
        redacted_pages.append(DocumentPage(number=page.number, text=text))

    residual_markers = re.compile(
        r"(?i)(?:Ф\.?И\.?О\.?|паспорт|СНИЛС|полис\s+ОМС|дата\s+рождения|адрес)\s*[:№]"
    )
    review_required = any(residual_markers.search(page.text) for page in redacted_pages)
    warnings: list[str] = []
    if review_required:
        warnings.append(
            "После автоматического обезличивания остались маркеры персональных данных; требуется ручная проверка."
        )

    report = PiiReport(
        replacement_counts=dict(sorted(counts.items())),
        total_replacements=sum(counts.values()),
        review_required=review_required,
        warnings=warnings,
    )
    return tuple(redacted_pages), report
