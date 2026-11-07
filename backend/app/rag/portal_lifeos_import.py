from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.documents.models import DocumentPage
from app.documents.redactor import redact_pages
from app.documents.security import scan_untrusted_text

SCHEMA_VERSION = 1
CONTENT_SCOPE = "reviewed_excerpt"
ALLOWED_SPECIALTIES = {"neurology", "neurosurgery"}
OUTPUT_PREFIX = "portal-article-"
MAX_BODY_CHARACTERS = 500_000


class PortalLifeOSImportError(ValueError):
    pass


def _required(record: dict[str, Any], key: str) -> str:
    value = str(record.get(key) or "").strip()
    if not value:
        raise PortalLifeOSImportError(f"отсутствует обязательное поле {key}")
    return value


def _iso_datetime(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PortalLifeOSImportError(f"{field} должен быть ISO datetime") from exc
    if parsed.tzinfo is None:
        raise PortalLifeOSImportError(f"{field} должен содержать timezone")
    return parsed


def _public_https(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise PortalLifeOSImportError("некорректный HTTPS URL") from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username
        or parsed.password
        or port not in (None, 443)
        or host == "localhost"
        or host.endswith(".local")
    ):
        raise PortalLifeOSImportError("источник должен быть публичным HTTPS URL")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address and not address.is_global:
        raise PortalLifeOSImportError("локальный адрес источника запрещён")
    return value


def _portal_url(value: str, source_id: str) -> str:
    value = _public_https(value)
    parsed = urlsplit(value)
    expected_slug = source_id.removeprefix(OUTPUT_PREFIX)
    if (
        (parsed.hostname or "").lower().rstrip(".") not in {"st8dom.ru", "www.st8dom.ru"}
        or parsed.path.rstrip("/") != f"/materials/{expected_slug}"
        or parsed.query
        or parsed.fragment
    ):
        raise PortalLifeOSImportError("portal_url не соответствует статье st8dom.ru")
    return value


def _validate_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("review_status") != "approved":
        raise PortalLifeOSImportError("review_status должен быть approved")
    if record.get("content_scope") != CONTENT_SCOPE:
        raise PortalLifeOSImportError(f"content_scope должен быть {CONTENT_SCOPE}")
    source_id = _required(record, "source_id").casefold()
    if not re.fullmatch(r"portal-article-[a-z0-9][a-z0-9-]{2,180}", source_id):
        raise PortalLifeOSImportError("некорректный source_id")
    title = _required(record, "title")
    raw_body = record.get("body")
    if not isinstance(raw_body, str) or not raw_body.strip():
        raise PortalLifeOSImportError("отсутствует обязательное поле body")
    # Hash the exact exported bytes. Normalising whitespace before verification
    # would make the integrity check reject a valid signed export (or verify text
    # different from the text that is ultimately stored).
    body = raw_body
    if not 100 <= len(body) <= MAX_BODY_CHARACTERS:
        raise PortalLifeOSImportError("текст статьи вне допустимого размера")
    expected_sha = _required(record, "body_sha256")
    actual_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or expected_sha != actual_sha:
        raise PortalLifeOSImportError("SHA-256 текста статьи не совпадает")
    reviewed_by = _required(record, "reviewed_by")
    reviewed_at = _required(record, "reviewed_at")
    _iso_datetime(reviewed_at, "reviewed_at")
    source_version = _required(record, "source_version")
    _iso_datetime(source_version, "source_version")
    published_at = _required(record, "published_at")
    published_datetime = _iso_datetime(published_at, "published_at")
    portal_url = _portal_url(_required(record, "portal_url"), source_id)
    specialties = tuple(
        sorted(set(record.get("specialty_tags") or []) & ALLOWED_SPECIALTIES)
    )
    if not specialties:
        raise PortalLifeOSImportError("нет профиля neurology/neurosurgery")
    raw_sources = record.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise PortalLifeOSImportError("отсутствуют проверяемые источники статьи")
    sources: list[dict[str, str]] = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise PortalLifeOSImportError("источник статьи должен быть объектом")
        sources.append(
            {
                "title": _required(raw, "title"),
                "url": _public_https(_required(raw, "url")),
            }
        )
    summary = str(record.get("summary") or "").strip()
    untrusted_content = "\n\n".join(
        [title, summary, body, *(source["title"] for source in sources)]
    )
    pages = (DocumentPage(number=1, text=untrusted_content),)
    if scan_untrusted_text(pages).prompt_injection_detected:
        raise PortalLifeOSImportError("обнаружены недоверенные инструкции")
    _, pii = redact_pages(pages)
    if pii.total_replacements or pii.review_required:
        raise PortalLifeOSImportError("обнаружены персональные идентификаторы")
    return {
        "source_id": source_id,
        "title": title,
        "summary": summary,
        "body": body,
        "body_sha256": actual_sha,
        "portal_url": portal_url,
        "source_version": source_version,
        "published_at": published_at,
        "publication_year": published_datetime.year,
        "reviewed_at": reviewed_at,
        "reviewed_by": reviewed_by,
        "author": str(record.get("author") or "").strip(),
        "rubric": str(record.get("rubric") or "").strip(),
        "specialty_tags": specialties,
        "tags": [str(tag) for tag in record.get("tags") or []],
        "sources": sources,
    }


def _frontmatter_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render(record: dict[str, Any]) -> str:
    fields: dict[str, Any] = {
        "title": record["title"],
        "year": record["publication_year"],
        "source": "Life-OS портала st8dom.ru (медицинская редакция)",
        "source_url": record["portal_url"],
        "url": record["portal_url"],
        "source_version": record["source_version"],
        "jurisdiction": "RU",
        "language": "ru",
        "document_type": "reviewed_lifeos_article",
        "specialty_tags": " ".join(record["specialty_tags"]),
        "content_scope": CONTENT_SCOPE,
        "review_status": "approved",
        "reviewed_by": record["reviewed_by"],
        "reviewed_at": record["reviewed_at"],
        "lifeos_source_id": record["source_id"],
        "lifeos_source_sha256": record["body_sha256"],
        "reference_urls": " ".join(source["url"] for source in record["sources"]),
        "full_text_checked": False,
    }
    frontmatter = "\n".join(
        f"{key}: {_frontmatter_value(value)}" for key, value in fields.items()
    )
    source_lines = "\n".join(
        f"- [{source['title']}]({source['url']})" for source in record["sources"]
    )
    return (
        f"---\n{frontmatter}\n---\n\n"
        f"# {record['title']}\n\n"
        "> Медицински отредактированная статья Life-OS. Это вторичный обзор: "
        "он не заменяет официальный полный текст клинической рекомендации.\n\n"
        f"{record['summary']}\n\n"
        f"{record['body'].rstrip()}\n\n"
        f"## Проверяемые источники статьи\n\n{source_lines}\n"
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def sync_portal_lifeos_export(export_path: Path, vitalis_vault: Path) -> dict[str, Any]:
    try:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PortalLifeOSImportError("не удалось прочитать export Life-OS") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PortalLifeOSImportError("неподдерживаемая версия export")
    if payload.get("source_system") != "st8dom-lifeos":
        raise PortalLifeOSImportError("неизвестная исходная система")
    if payload.get("automatic_ingestion") is not False:
        raise PortalLifeOSImportError("automatic_ingestion должен быть false")
    raw_records = payload.get("approved_records")
    if not isinstance(raw_records, list):
        raise PortalLifeOSImportError("approved_records должен быть массивом")

    accepted: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    duplicate_ids: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            rejected.append({"record": str(index), "reason": "запись не является объектом"})
            continue
        try:
            record = _validate_record(raw)
        except PortalLifeOSImportError as exc:
            rejected.append(
                {"record": str(raw.get("source_id") or index), "reason": str(exc)}
            )
            continue
        if record["source_id"] in accepted:
            duplicate_ids.add(record["source_id"])
        else:
            accepted[record["source_id"]] = record
    for source_id in duplicate_ids:
        accepted.pop(source_id, None)
        rejected.append({"record": source_id, "reason": "дублирующийся source_id"})

    target = vitalis_vault / "generated" / "lifeos" / "portal"
    expected: set[str] = set()
    manifest_records: list[dict[str, Any]] = []
    for source_id, record in sorted(accepted.items()):
        filename = f"{source_id}.md"
        expected.add(filename)
        _atomic_text(target / filename, _render(record))
        manifest_records.append(
            {
                "source_id": source_id,
                "title": record["title"],
                "source_url": record["portal_url"],
                "source_version": record["source_version"],
                "source_sha256": record["body_sha256"],
                "output": filename,
                "reviewed_by": record["reviewed_by"],
                "reviewed_at": record["reviewed_at"],
                "content_scope": CONTENT_SCOPE,
            }
        )
    removed = 0
    for path in target.glob(f"{OUTPUT_PREFIX}*.md"):
        if path.name not in expected:
            path.unlink()
            removed += 1
    candidates = payload.get("candidates")
    candidate_rows = candidates if isinstance(candidates, list) else []
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_export": str(export_path),
        "automatic_ingestion": False,
        "review_required": True,
        "imported": len(manifest_records),
        "rejected": rejected,
        "candidate_count": len(candidate_rows),
        "candidates": [
            {
                "source_id": row.get("source_id"),
                "title": row.get("title"),
                "review_status": "needs_medical_review",
            }
            for row in candidate_rows
            if isinstance(row, dict)
        ],
        "records": manifest_records,
    }
    _atomic_text(
        target / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "approved_in_export": len(raw_records),
        "imported": len(manifest_records),
        "rejected": len(rejected),
        "candidates": len(candidate_rows),
        "removed": removed,
        "manifest": str(target / "manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Импорт проверенного read-only export Life-OS портала в Vitalis"
    )
    parser.add_argument("export", type=Path)
    parser.add_argument("vitalis_vault", type=Path)
    args = parser.parse_args()
    stats = sync_portal_lifeos_export(args.export, args.vitalis_vault)
    print(json.dumps(stats, ensure_ascii=False))
    return 1 if stats["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
