from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ALLOWED_OFFICIAL_DOMAINS = (
    "cr.minzdrav.gov.ru",
    "escardio.org",
    "nice.org.uk",
    "who.int",
)
VALID_STATUSES = {"needs_review", "approved", "rejected", "superseded"}
VALID_CONTENT_SCOPES = {
    "bibliographic_metadata_only",
    "reviewed_excerpt",
    "reviewed_full_text",
}
REQUIRED_FIELDS = {
    "id",
    "title",
    "url",
    "organization",
    "jurisdiction",
    "language",
    "document_type",
    "content_scope",
    "publication_year",
    "source_version",
    "checked_at",
    "status",
    "ingest_allowed",
    "topics",
    "verification_level",
    "review_notes",
}


def _is_official_https_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username
        or parsed.password
        or port not in (None, 443)
    ):
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in ALLOWED_OFFICIAL_DOMAINS)


def load_registry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if registry.get("default_ingestion_policy") != "deny":
        errors.append("default_ingestion_policy must be deny")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        return errors + ["sources must be a non-empty list"]

    identifiers: set[str] = set()
    current_year = date.today().year
    for index, source in enumerate(sources):
        prefix = f"source[{index}]"
        if not isinstance(source, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = sorted(REQUIRED_FIELDS - set(source))
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")
            continue
        source_id = source["id"]
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{prefix}.id must be a non-empty string")
        elif source_id in identifiers:
            errors.append(f"duplicate id: {source_id}")
        identifiers.add(source_id)
        if not _is_official_https_url(str(source["url"])):
            errors.append(f"{source_id}: URL is not an allowlisted official HTTPS URL")
        if source["status"] not in VALID_STATUSES:
            errors.append(f"{source_id}: invalid status")
        if source["content_scope"] not in VALID_CONTENT_SCOPES:
            errors.append(f"{source_id}: invalid content_scope")
        if not isinstance(source["ingest_allowed"], bool):
            errors.append(f"{source_id}: ingest_allowed must be boolean")
        if source["ingest_allowed"] and source["status"] != "approved":
            errors.append(f"{source_id}: only approved sources may be ingested")
        if source["status"] == "approved":
            for field in ("reviewed_at", "reviewed_by"):
                if not source.get(field):
                    errors.append(f"{source_id}: approved source requires {field}")
        if (
            source["ingest_allowed"]
            and source["content_scope"] == "bibliographic_metadata_only"
            and source["document_type"] != "clinical_guideline_metadata"
        ):
            errors.append(
                f"{source_id}: metadata-only ingestion requires "
                "document_type=clinical_guideline_metadata"
            )
        year = source["publication_year"]
        if not isinstance(year, int) or not 1900 <= year <= current_year + 1:
            errors.append(f"{source_id}: invalid publication_year")
        try:
            date.fromisoformat(str(source["checked_at"]))
        except ValueError:
            errors.append(f"{source_id}: checked_at must be ISO date")
        topics = source["topics"]
        if not isinstance(topics, list) or not topics or not all(
            isinstance(topic, str) and topic for topic in topics
        ):
            errors.append(f"{source_id}: topics must be a non-empty string list")
        if not str(source["review_notes"]).strip():
            errors.append(f"{source_id}: review_notes must explain the review state")
    return errors


def approved_for_ingestion(registry: dict[str, Any]) -> list[dict[str, Any]]:
    errors = validate_registry(registry)
    if errors:
        raise ValueError("invalid source registry: " + "; ".join(errors))
    return [
        source
        for source in registry["sources"]
        if source["status"] == "approved" and source["ingest_allowed"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Vitalis source registry")
    parser.add_argument("command", choices=["validate", "list-approved"])
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    errors = validate_registry(registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    approved = approved_for_ingestion(registry)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "ok": True,
                    "sources": len(registry["sources"]),
                    "approved_for_ingestion": len(approved),
                },
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(approved, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
