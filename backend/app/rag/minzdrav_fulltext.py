from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from app.documents.isolated_parser import extract_pdf_isolated
from app.documents.models import DocumentPage, ExtractedDocument
from app.documents.security import scan_untrusted_text
from app.rag.portal_guidelines import select_portal_guidelines
from app.rag.safe_fetch import fetch_bounded

API_HOST = "apicr.minzdrav.gov.ru"
API_PATH = "/API.ashx"
API_OPERATION = "GetClinrecPdf"
CONTENT_SCOPE = "official_full_text_extracted"
GENERATED_PREFIX = "minzdrav-full-"
MANIFEST_SCHEMA_VERSION = 1
MAX_PDF_BYTES = 8_000_000
MAX_PAGES = 400
MAX_CHARACTERS = 1_500_000
FETCH_TIMEOUT_SECONDS = 45.0


class MinzdravFullTextError(ValueError):
    pass


def official_document_url(external_id: str) -> str:
    if not re.fullmatch(r"\d+_\d+", external_id):
        raise MinzdravFullTextError("Недопустимый идентификатор КР Минздрава")
    return f"https://{API_HOST}{API_PATH}?{urlencode({'op': API_OPERATION, 'id': external_id})}"


def is_allowed_document_url(url: str, external_id: str | None = None) -> bool:
    try:
        parsed = urlsplit(url)
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        port = parsed.port
    except (ValueError, TypeError):
        return False
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower().rstrip(".") != API_HOST
        or parsed.username
        or parsed.password
        or port not in (None, 443)
        or parsed.path != API_PATH
        or parsed.fragment
        or set(query) != {"op", "id"}
        or query.get("op") != [API_OPERATION]
        or len(query.get("id", [])) != 1
        or not re.fullmatch(r"\d+_\d+", query["id"][0])
    ):
        return False
    return external_id is None or query["id"] == [external_id]


def _normalised_title_tokens(value: str) -> set[str]:
    stop = {
        "болезнь",
        "синдром",
        "детей",
        "взрослых",
        "система",
        "нервной",
        "головного",
        "мозга",
        "ассоциированный",
    }
    tokens: set[str] = set()
    for token in re.findall(r"[a-zа-яё0-9]+", value.casefold().replace("ё", "е")):
        if len(token) < 4 or token in stop:
            continue
        # Prefix stems tolerate common Russian case and gender endings without a
        # language model or an external morphology dependency.
        tokens.add(token if len(token) <= 5 else token[:6])
    return tokens


def _title_matches(title: str, pages: tuple[DocumentPage, ...]) -> bool:
    expected = _normalised_title_tokens(title)
    if not expected:
        return False
    observed = _normalised_title_tokens(
        "\n".join(page.text for page in pages[:20])
    )
    matched = expected & observed
    minimum = 1 if len(expected) == 1 else max(2, (len(expected) + 1) // 2)
    return len(matched) >= minimum


def _normalise_code(value: str) -> str:
    return re.sub(r"\s+", "", value.upper().replace("А", "A"))


def _code_matches(codes: list[str], pages: tuple[DocumentPage, ...]) -> bool:
    if not codes:
        return False
    text = "\n".join(page.text for page in pages[:45]).upper().replace("А", "A")
    return any(
        re.search(rf"(?<![A-ZА-Я0-9]){re.escape(_normalise_code(code))}(?![0-9])", text)
        for code in codes
    )


def validate_extracted_document(
    item: dict[str, Any], document: ExtractedDocument
) -> None:
    if len(document.pages) < 2 or document.character_count < 1_000:
        raise MinzdravFullTextError(
            f"{item['external_id']}: извлечённый документ слишком короткий"
        )
    if not _title_matches(str(item["title"]), document.pages):
        raise MinzdravFullTextError(
            f"{item['external_id']}: название КР не подтверждено в PDF"
        )
    if not _code_matches(list(item.get("codes") or []), document.pages):
        raise MinzdravFullTextError(
            f"{item['external_id']}: код МКБ не подтверждён в PDF"
        )
    security = scan_untrusted_text(document.pages)
    if security.prompt_injection_detected:
        kinds = ", ".join(warning.kind for warning in security.warnings)
        raise MinzdravFullTextError(
            f"{item['external_id']}: обнаружены недоверенные инструкции ({kinds})"
        )


async def download_and_extract(item: dict[str, Any]) -> tuple[ExtractedDocument, str]:
    external_id = str(item["external_id"])
    url = official_document_url(external_id)
    try:
        data, content_type, final_url, _ = await fetch_bounded(
            url,
            allowed=lambda candidate: is_allowed_document_url(candidate, external_id),
            max_bytes=MAX_PDF_BYTES,
            timeout=FETCH_TIMEOUT_SECONDS,
            user_agent="Vitalis-Clinical-Guideline-Sync/1.0",
        )
    except Exception as exc:
        raise MinzdravFullTextError(
            f"{external_id}: не удалось получить официальный PDF"
        ) from exc
    if not is_allowed_document_url(final_url, external_id):
        raise MinzdravFullTextError(f"{external_id}: неожиданный конечный URL")
    if "pdf" not in content_type and content_type not in {
        "application/octet-stream",
        "binary/octet-stream",
        "",
    }:
        raise MinzdravFullTextError(
            f"{external_id}: официальный endpoint вернул не PDF"
        )
    if not data.startswith(b"%PDF-"):
        raise MinzdravFullTextError(f"{external_id}: сигнатура ответа не соответствует PDF")
    try:
        document = await extract_pdf_isolated(
            data,
            max_pages=MAX_PAGES,
            max_chars=MAX_CHARACTERS,
            timeout=40,
        )
    except Exception as exc:
        raise MinzdravFullTextError(
            f"{external_id}: безопасное извлечение PDF не удалось"
        ) from exc
    validate_extracted_document(item, document)
    return document, final_url


def _frontmatter_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_full_text_document(
    item: dict[str, Any],
    document: ExtractedDocument,
    document_url: str,
    *,
    authorized_by: str,
    authorized_at: str,
) -> str:
    codes = ", ".join(item.get("codes") or [])
    specialties = " ".join(item.get("specialty_tags") or [])
    fields: dict[str, Any] = {
        "title": item["title"],
        "mkb": (item.get("codes") or [""])[0],
        "mkb_codes": codes,
        "year": item["publication_year"],
        "published_on": item["published_on"],
        "source": "Минздрав России",
        "organization": item["organization"],
        "jurisdiction": "RU",
        "language": "ru",
        "document_type": "clinical_guideline_full_text",
        "source_version": item["source_version"],
        "source_record_id": item["external_id"],
        "guideline_status": "active",
        "current_version_checked_at": authorized_at,
        "url": item["url"],
        "source_url": item["url"],
        "document_url": document_url,
        "specialty_tags": specialties,
        "content_scope": CONTENT_SCOPE,
        "review_status": "approved_official_extraction",
        "authorized_by": authorized_by,
        "authorized_at": authorized_at,
        "full_text_checked": True,
        "page_count": len(document.pages),
        "character_count": document.character_count,
        "full_text_sha256": document.sha256,
    }
    frontmatter = "\n".join(
        f"{key}: {_frontmatter_value(value)}" for key, value in fields.items()
    )
    body = [
        f"# {item['title']}",
        "",
        "> Текст автоматически извлечён из официального PDF Минздрава России. "
        "Он используется как проверяемый первичный источник с указанием страниц; "
        "клиническое решение принимает врач.",
    ]
    for page in document.pages:
        body.extend(("", f"## Страница {page.number}", "", page.text or "[Пустая страница]"))
    return f"---\n{frontmatter}\n---\n\n" + "\n".join(body).rstrip() + "\n"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "records": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MinzdravFullTextError("Существующий manifest.json повреждён") from exc
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise MinzdravFullTextError("Неподдерживаемая версия manifest.json")
    return value


async def sync_full_texts(
    portal_registry_path: Path,
    vault_dir: Path,
    *,
    authorized_by: str,
    authorized_at: str,
) -> dict[str, Any]:
    try:
        date.fromisoformat(authorized_at)
    except ValueError as exc:
        raise MinzdravFullTextError("authorized_at должен быть датой ISO") from exc
    if not authorized_by.strip():
        raise MinzdravFullTextError("authorized_by не может быть пустым")

    registry = json.loads(portal_registry_path.read_text(encoding="utf-8"))
    selected = select_portal_guidelines(registry)
    target_dir = vault_dir / "generated" / "guidelines" / "rf"
    manifest_path = target_dir / "manifest.json"
    previous = _load_manifest(manifest_path)
    records = {
        str(record.get("external_id")): dict(record)
        for record in previous.get("records", [])
        if record.get("external_id")
    }
    failures: list[dict[str, str]] = []
    written = 0
    expected_names = {
        f"{GENERATED_PREFIX}{item['external_id']}.md" for item in selected
    }

    for item in selected:
        external_id = str(item["external_id"])
        filename = f"{GENERATED_PREFIX}{external_id}.md"
        try:
            document, document_url = await download_and_extract(item)
            rendered = render_full_text_document(
                item,
                document,
                document_url,
                authorized_by=authorized_by,
                authorized_at=authorized_at,
            )
            _atomic_text(target_dir / filename, rendered)
            records[external_id] = {
                "external_id": external_id,
                "title": item["title"],
                "source_version": item["source_version"],
                "guideline_status": "active",
                "current_version_checked_at": authorized_at,
                "source_url": item["url"],
                "document_url": document_url,
                "output": filename,
                "sha256": document.sha256,
                "page_count": len(document.pages),
                "character_count": document.character_count,
                "content_scope": CONTENT_SCOPE,
                "authorized_by": authorized_by,
                "authorized_at": authorized_at,
            }
            written += 1
        except Exception as exc:
            failures.append({"external_id": external_id, "error": str(exc)})

    removed = 0
    for path in target_dir.glob(f"{GENERATED_PREFIX}*.md"):
        if path.name not in expected_names:
            path.unlink()
            removed += 1
    selected_ids = {str(item["external_id"]) for item in selected}
    records = {
        external_id: record
        for external_id, record in records.items()
        if external_id in selected_ids and (target_dir / str(record.get("output"))).is_file()
    }
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": now,
        "source_registry": str(portal_registry_path),
        "sync_status": "complete" if not failures else "partial",
        "selected": len(selected),
        "successful_records": len(records),
        "failures": failures,
        "records": [records[key] for key in sorted(records)],
    }
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {
        "selected": len(selected),
        "written": written,
        "available": len(records),
        "removed": removed,
        "failed": len(failures),
        "manifest": str(manifest_path),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Безопасная загрузка полных текстов КР из официального API Минздрава"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync = subparsers.add_parser("sync")
    sync.add_argument("portal_registry", type=Path)
    sync.add_argument("vault_dir", type=Path)
    sync.add_argument("--authorized-by", required=True)
    sync.add_argument("--authorized-at", required=True)
    args = parser.parse_args()
    stats = asyncio.run(
        sync_full_texts(
            args.portal_registry,
            args.vault_dir,
            authorized_by=args.authorized_by,
            authorized_at=args.authorized_at,
        )
    )
    print(json.dumps(stats, ensure_ascii=False))
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
