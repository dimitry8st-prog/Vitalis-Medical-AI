from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.rag.source_policy import is_primary_evidence_source

URL_RE = re.compile(r"https?://[^\s<>()\[\]{}]+", re.IGNORECASE)
DOSE_RE = re.compile(
    r"(?i)(?<!\w)\d+(?:[.,]\d+)?(?:\s*[–—-]\s*\d+(?:[.,]\d+)?)?\s*"
    r"(?:мг|mg|мкг|mcg|µg|г|g|мл|ml|ед|iu)"
    r"(?:\s*/\s*(?:сут(?:ки)?|day|день|кг|kg))?\b"
)
UNVERIFIED_URL_MARKER = "[непроверенная ссылка удалена]"
UNSUPPORTED_DOSE_MARKER = (
    "Точная дозировка скрыта: она не подтверждена переданным полным текстом источника."
)


@dataclass(frozen=True)
class CitationGuardReport:
    cited_urls: int
    verified_urls: int
    removed_urls: int
    dose_statements: int
    supported_dose_statements: int
    removed_dose_statements: int
    missing_required_citation: bool
    blocked: bool
    entailment_verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "cited_urls": self.cited_urls,
            "verified_urls": self.verified_urls,
            "removed_urls": self.removed_urls,
            "dose_statements": self.dose_statements,
            "supported_dose_statements": self.supported_dose_statements,
            "removed_dose_statements": self.removed_dose_statements,
            "missing_required_citation": self.missing_required_citation,
            "blocked": self.blocked,
            "entailment_verified": self.entailment_verified,
            "manual_review_required": True,
        }


def _normalize_url(value: str) -> str:
    value = value.rstrip(".,;:!?\"'")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    host = parsed.hostname.lower().rstrip(".")
    netloc = host
    if port and port not in {80, 443}:
        netloc = f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def _allowed_urls(web_hits: list[dict[str, Any]], rag_hits: list[dict[str, Any]]) -> set[str]:
    allowed: set[str] = set()
    for hit in web_hits:
        if hit.get("content_scope") == "bibliographic_metadata_only":
            continue
        normalized = _normalize_url(str(hit.get("url") or ""))
        if normalized:
            allowed.add(normalized)
    for hit in rag_hits:
        meta = hit.get("meta") or {}
        if meta.get("content_scope") == "bibliographic_metadata_only":
            continue
        for key in ("url", "source_url"):
            normalized = _normalize_url(str(meta.get(key) or ""))
            if normalized:
                allowed.add(normalized)
    return allowed


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"true", "yes", "1", "да"}


def _supported_doses(
    web_hits: list[dict[str, Any]], rag_hits: list[dict[str, Any]]
) -> set[str]:
    supported: set[str] = set()
    for hit in web_hits:
        if not hit.get("full_text_checked") or not is_primary_evidence_source(
            str(hit.get("url") or "")
        ):
            continue
        evidence = " ".join(
            str(hit.get(key) or "")
            for key in ("evidence_excerpt", "full_text_excerpt")
        )
        supported.update(_normalize_dose(match.group(0)) for match in DOSE_RE.finditer(evidence))
    for hit in rag_hits:
        meta = hit.get("meta") or {}
        source_url = str(meta.get("source_url") or meta.get("url") or "")
        if (
            meta.get("content_scope") != "official_full_text_extracted"
            or not _truthy(meta.get("full_text_checked"))
            or not is_primary_evidence_source(source_url)
        ):
            continue
        evidence = str(hit.get("text") or "")
        supported.update(
            _normalize_dose(match.group(0)) for match in DOSE_RE.finditer(evidence)
        )
    return supported


def _normalize_dose(value: str) -> str:
    return re.sub(r"\s+", "", value.lower().replace(",", ".").replace("—", "–"))


def guard_clinical_payload(
    payload: dict[str, Any] | None,
    *,
    web_hits: list[dict[str, Any]],
    rag_hits: list[dict[str, Any]],
    strict: bool,
) -> dict[str, Any] | None:
    if not payload or not payload.get("content"):
        return payload

    content = str(payload["content"])
    allowed = _allowed_urls(web_hits, rag_hits)
    supported_doses = _supported_doses(web_hits, rag_hits)
    cited_urls = 0
    verified_urls = 0
    removed_urls = 0

    def replace_url(match: re.Match[str]) -> str:
        nonlocal cited_urls, verified_urls, removed_urls
        cited_urls += 1
        raw = match.group(0)
        trailing = raw[len(raw.rstrip(".,;:!?\"'")) :]
        if _normalize_url(raw) in allowed:
            verified_urls += 1
            return raw
        removed_urls += 1
        return UNVERIFIED_URL_MARKER + trailing

    content = URL_RE.sub(replace_url, content)
    dose_statements = 0
    supported_dose_statements = 0
    removed_dose_statements = 0
    guarded_lines: list[str] = []
    for line in content.splitlines():
        doses = [_normalize_dose(match.group(0)) for match in DOSE_RE.finditer(line)]
        if not doses:
            guarded_lines.append(line)
            continue
        dose_statements += 1
        if all(dose in supported_doses for dose in doses):
            supported_dose_statements += 1
            guarded_lines.append(line)
        else:
            removed_dose_statements += 1
            prefix = "- " if line.lstrip().startswith(('-', '*')) else ""
            guarded_lines.append(prefix + UNSUPPORTED_DOSE_MARKER)

    missing_required_citation = bool(allowed) and cited_urls == 0
    blocked = strict and (
        removed_urls > 0
        or removed_dose_statements > 0
        or missing_required_citation
    )
    if blocked:
        content = (
            "Ответ заблокирован: модель сформировала ссылку или точную дозировку, "
            "которые не подтверждены переданным проверенным источником. Повторите поиск "
            "или проверьте первоисточник вручную."
        )
    else:
        content = "\n".join(guarded_lines)
        warnings: list[str] = []
        if removed_urls:
            warnings.append("Непроверенные ссылки модели удалены.")
        if removed_dose_statements:
            warnings.append("Неподтверждённые точные дозировки скрыты.")
        if missing_required_citation:
            warnings.append("Ответ модели не содержит проверяемой ссылки из контекста.")
        if warnings:
            content = content.rstrip() + "\n\n## Ограничения автоматической проверки\n\n" + " ".join(warnings)

    report = CitationGuardReport(
        cited_urls=cited_urls,
        verified_urls=verified_urls,
        removed_urls=removed_urls,
        dose_statements=dose_statements,
        supported_dose_statements=supported_dose_statements,
        removed_dose_statements=removed_dose_statements,
        missing_required_citation=missing_required_citation,
        blocked=blocked,
    )
    guarded = dict(payload)
    guarded["content"] = content
    guarded["citation_validation"] = report.as_dict()
    return guarded
