from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime, timezone
from typing import Any

from app.rag.source_policy import is_primary_guideline_source
from app.services.citation_guard import DOSE_RE

STATUS_CHECK_MAX_AGE_DAYS = 45
_PAGE_RE = re.compile(r"\[Страница\s+(\d+)\]", re.IGNORECASE)
_ROUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("внутрь", re.compile(r"\b(?:внутрь|перорально|per\s+os|p\.?\s*o\.?)\b", re.I)),
    ("внутривенно", re.compile(r"\b(?:внутривенно|в/в|intravenous(?:ly)?|i\.?\s*v\.?)\b", re.I)),
    ("внутримышечно", re.compile(r"\b(?:внутримышечно|в/м|intramuscular(?:ly)?|i\.?\s*m\.?)\b", re.I)),
    ("подкожно", re.compile(r"\b(?:подкожно|п/к|subcutaneous(?:ly)?|s\.?\s*c\.?)\b", re.I)),
    ("интраназально", re.compile(r"\b(?:интраназально|intranasal(?:ly)?)\b", re.I)),
    ("ректально", re.compile(r"\b(?:ректально|rectal(?:ly)?)\b", re.I)),
    ("ингаляционно", re.compile(r"\b(?:ингаляционно|inhaled|inhalation)\b", re.I)),
)


def build_medication_card(
    rag_hits: list[dict[str, Any]],
    web_hits: list[dict[str, Any]],
    *,
    lang: str,
    as_of: date | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    """Build a medication block only from current, verified guideline excerpts.

    The model never supplies these rows. Every displayed regimen is a literal excerpt
    from an allowlisted full guideline and carries enough provenance for a clinician
    to reopen the source.
    """

    today = as_of or datetime.now(timezone.utc).date()
    items: list[dict[str, Any]] = []
    blocked_statements = 0
    seen: set[tuple[str, str]] = set()

    for source in _iter_sources(rag_hits, web_hits, lang=lang):
        statements = list(_dose_statements(source["text"]))
        if not statements:
            continue
        current, _ = _current_source_state(source, today)
        if not current:
            blocked_statements += len(statements)
            continue
        for statement in statements:
            if source["requires_page"] and statement["page"] is None:
                blocked_statements += 1
                continue
            dedupe = (
                re.sub(r"\s+", " ", statement["statement"]).casefold(),
                source["source_url"].rstrip("/").casefold(),
            )
            if dedupe in seen:
                continue
            seen.add(dedupe)
            items.append(
                {
                    "regimen": statement["statement"],
                    "doses": statement["doses"],
                    "routes": _routes(statement["statement"]),
                    "page": statement["page"],
                    "source_title": source["source_title"],
                    "source_url": source["source_url"],
                    "source_year": source["source_year"],
                    "source_version": source["source_version"],
                    "source_record_id": source["source_record_id"],
                    "jurisdiction": source["jurisdiction"],
                    "version_checked_at": source["version_checked_at"],
                    "evidence_quote": statement["statement"],
                    "literal_source_match": True,
                }
            )
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break

    if items:
        status = "verified"
        message = (
            "Показаны только дословные режимы из полного текста действующей версии "
            "клинической рекомендации. Не пересчитывайте дозу без проверки контекста, "
            "показаний и характеристик пациента."
        )
    elif blocked_statements:
        status = "blocked_unconfirmed_current_version"
        message = (
            "В источниках обнаружены дозировки, но карточка скрыла их: статус действующей "
            "версии, дата его проверки или точный локатор не соответствуют правилу "
            "актуальности Vitalis."
        )
    else:
        status = "not_found"
        message = (
            "В найденных проверенных фрагментах действующей клинической рекомендации "
            "препараты с точными дозировками не обнаружены."
        )

    return {
        "status": status,
        "as_of": today.isoformat(),
        "status_check_max_age_days": STATUS_CHECK_MAX_AGE_DAYS,
        "items": items,
        "blocked_statements": blocked_statements,
        "message": message,
        "medical_decision_required": True,
    }


def is_current_guideline_metadata(
    meta: dict[str, Any],
    *,
    as_of: date | None = None,
) -> bool:
    """Age alone never invalidates a guideline whose active status was rechecked."""

    today = as_of or datetime.now(timezone.utc).date()
    source = {
        "guideline_status": meta.get("guideline_status"),
        "version_checked_at": (
            meta.get("current_version_checked_at") or meta.get("authorized_at")
        ),
        "source_version": meta.get("source_version"),
    }
    return _current_source_state(source, today)[0]


def _iter_sources(
    rag_hits: list[dict[str, Any]],
    web_hits: list[dict[str, Any]],
    *,
    lang: str,
) -> Iterable[dict[str, Any]]:
    for hit in rag_hits:
        meta = hit.get("meta") or {}
        source_url = str(meta.get("source_url") or meta.get("url") or "")
        jurisdiction = str(meta.get("jurisdiction") or "")
        language = str(meta.get("language") or "").casefold()
        if lang == "ru" and language not in {"", "ru", "rus", "russian"}:
            continue
        if lang == "en" and (
            language in {"ru", "rus", "russian"}
            or jurisdiction.casefold() in {"ru", "russia", "россия"}
        ):
            continue
        if (
            meta.get("content_scope") != "official_full_text_extracted"
            or not _truthy(meta.get("full_text_checked"))
            or not is_primary_guideline_source(source_url)
        ):
            continue
        yield {
            "text": str(hit.get("text") or ""),
            "source_title": str(hit.get("title") or meta.get("title") or source_url),
            "source_url": source_url,
            "source_year": _year(hit.get("year") or meta.get("year")),
            "source_version": str(meta.get("source_version") or ""),
            "source_record_id": str(meta.get("source_record_id") or ""),
            "jurisdiction": jurisdiction or ("RU" if lang == "ru" else "international"),
            "guideline_status": str(meta.get("guideline_status") or ""),
            "version_checked_at": str(
                meta.get("current_version_checked_at")
                or meta.get("authorized_at")
                or ""
            ),
            "stale": bool(hit.get("stale")),
            "requires_page": True,
        }

    for hit in web_hits:
        if str(hit.get("lang") or "").casefold() != lang:
            continue
        source_url = str(hit.get("url") or "")
        if (
            not _truthy(hit.get("full_text_checked"))
            or not is_primary_guideline_source(source_url)
        ):
            continue
        evidence = str(hit.get("evidence_excerpt") or hit.get("full_text_excerpt") or "")
        if not evidence:
            continue
        yield {
            "text": evidence,
            "source_title": str(hit.get("title") or source_url),
            "source_url": source_url,
            "source_year": _year(hit.get("year")),
            "source_version": str(hit.get("source_version") or ""),
            "source_record_id": str(hit.get("source_record_id") or ""),
            "jurisdiction": str(hit.get("jurisdiction") or ("RU" if lang == "ru" else "international")),
            "guideline_status": str(hit.get("guideline_status") or ""),
            "version_checked_at": str(
                hit.get("current_version_checked_at")
                or hit.get("version_checked_at")
                or ""
            ),
            "stale": bool(hit.get("stale")),
            "requires_page": False,
        }


def _current_source_state(source: dict[str, Any], today: date) -> tuple[bool, str]:
    if str(source.get("guideline_status") or "").casefold() not in {
        "active",
        "current",
    }:
        return False, "current_version_not_confirmed"
    checked = _iso_date(source.get("version_checked_at"))
    if checked is None:
        return False, "version_check_date_missing"
    age = (today - checked).days
    if age < -1:
        return False, "version_check_date_in_future"
    if age > STATUS_CHECK_MAX_AGE_DAYS:
        return False, "version_check_expired"
    if not source.get("source_version"):
        return False, "source_version_missing"
    return True, "current"


def _dose_statements(text: str) -> Iterable[dict[str, Any]]:
    page_match = _PAGE_RE.search(text or "")
    page = int(page_match.group(1)) if page_match else None
    clean = _PAGE_RE.sub("", text or "")
    parts = re.split(r"(?<=[.!?;])\s+|\n+", clean)
    for raw in parts:
        statement = re.sub(r"\s+", " ", raw).strip(" \t\r\n-*•")
        if len(statement) < 8:
            continue
        doses = list(dict.fromkeys(match.group(0).strip() for match in DOSE_RE.finditer(statement)))
        if not doses:
            continue
        if len(statement) > 900:
            first = next(DOSE_RE.finditer(statement))
            start = max(0, first.start() - 320)
            end = min(len(statement), first.end() + 420)
            statement = statement[start:end].strip()
            doses = list(dict.fromkeys(match.group(0).strip() for match in DOSE_RE.finditer(statement)))
        yield {"statement": statement, "doses": doses, "page": page}


def _routes(statement: str) -> list[str]:
    return [label for label, pattern in _ROUTE_PATTERNS if pattern.search(statement)]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"true", "yes", "1", "да"}


def _year(value: Any) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _iso_date(value: Any) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None
