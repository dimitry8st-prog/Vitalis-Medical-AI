from __future__ import annotations

import re

from app.documents.models import DocumentPage, SecurityReport, SecurityWarning

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(r"(?i)(?:ignore|disregard).{0,40}(?:instruction|prompt|rule)"),
    ),
    (
        "ignore_instructions_ru",
        re.compile(r"(?i)игнорир(?:уй|овать).{0,40}(?:инструкц|правил|промпт)"),
    ),
    ("system_prompt", re.compile(r"(?i)(?:system|developer)\s+(?:prompt|message)")),
    (
        "system_prompt_ru",
        re.compile(r"(?i)(?:системн|скрыт).{0,30}(?:промпт|инструкц)"),
    ),
    (
        "execute_command",
        re.compile(
            r"(?i)(?:execute|run|выполни|запусти).{0,30}(?:command|script|команд|скрипт)"
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"(?i)(?:reveal|send|покажи|отправь).{0,40}(?:secret|token|ключ|секрет)"
        ),
    ),
)


def scan_untrusted_text(pages: tuple[DocumentPage, ...]) -> SecurityReport:
    warnings: list[SecurityWarning] = []
    seen: set[tuple[str, int]] = set()
    for page in pages:
        for kind, pattern in _INJECTION_PATTERNS:
            if pattern.search(page.text) and (kind, page.number) not in seen:
                seen.add((kind, page.number))
                warnings.append(SecurityWarning(kind=kind, page=page.number))
    return SecurityReport(
        prompt_injection_detected=bool(warnings),
        warnings=warnings,
    )
