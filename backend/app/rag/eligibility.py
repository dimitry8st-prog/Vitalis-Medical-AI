from __future__ import annotations

from typing import Any, Mapping

CLINICAL_GUIDELINE_DOCUMENT_TYPES = {
    "clinical_guideline_full_text",
}

CLINICAL_GUIDELINE_CONTENT_SCOPES = {
    "official_full_text_extracted",
}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def is_clinical_guideline_meta(meta: Mapping[str, Any] | None) -> bool:
    """Deny by default: only verified full clinical guidelines enter RAG."""
    if not meta:
        return False
    return (
        str(meta.get("document_type") or "")
        in CLINICAL_GUIDELINE_DOCUMENT_TYPES
        and str(meta.get("content_scope") or "")
        in CLINICAL_GUIDELINE_CONTENT_SCOPES
        and _truthy(meta.get("full_text_checked"))
    )


def is_clinical_guideline_web_hit(hit: Mapping[str, Any] | None) -> bool:
    """Web results are answer evidence only when classified as a guideline."""
    return bool(hit) and str(hit.get("document_type") or "") == "clinical_guideline"


def is_verified_full_guideline_web_hit(hit: Mapping[str, Any] | None) -> bool:
    """Only successfully retrieved guideline text can enter the answer context."""
    return (
        is_clinical_guideline_web_hit(hit)
        and _truthy(hit.get("full_text_checked"))
        and bool(str(hit.get("evidence_excerpt") or "").strip())
    )
