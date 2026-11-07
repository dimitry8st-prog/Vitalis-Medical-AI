from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.documents.models import DocumentPage
from app.documents.redactor import redact_pages
from app.documents.security import scan_untrusted_text

CONTENT_SCOPE = "reviewed_excerpt"
ALLOWED_SPECIALTIES = {"neurology", "neurosurgery"}
GENERATED_PREFIX = "lifeos-"
MANIFEST_SCHEMA_VERSION = 1
MAX_NOTE_BYTES = 2_000_000
MAX_NOTE_CHARACTERS = 500_000


class LifeOSImportError(ValueError):
    pass


@dataclass(frozen=True)
class ApprovedNote:
    path: Path
    relative_path: str
    source_id: str
    title: str
    source_url: str
    source_version: str
    year: int
    jurisdiction: str
    language: str
    specialty_tags: tuple[str, ...]
    reviewed_by: str
    reviewed_at: str
    body: str
    sha256: str


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        raise LifeOSImportError("отсутствует YAML-frontmatter")
    parts = raw.split("---", 2)
    if len(parts) != 3:
        raise LifeOSImportError("frontmatter не закрыт")
    meta: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise LifeOSImportError("frontmatter содержит неподдерживаемую строку")
        key, value = line.split(":", 1)
        key = key.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise LifeOSImportError(f"недопустимое поле frontmatter: {key}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            try:
                value = str(json.loads(value))
            except (TypeError, ValueError) as exc:
                raise LifeOSImportError(f"поле {key} содержит некорректную строку") from exc
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        meta[key] = value
    return meta, parts[2].strip()


def _required(meta: dict[str, str], key: str) -> str:
    value = re.sub(r"\s+", " ", str(meta.get(key) or "")).strip()
    if not value:
        raise LifeOSImportError(f"отсутствует обязательное поле {key}")
    return value


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"true", "yes", "1", "да"}


def _specialties(value: str) -> tuple[str, ...]:
    tags = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]*", value.casefold())
        if token in ALLOWED_SPECIALTIES
    }
    if not tags:
        raise LifeOSImportError(
            "specialty_tags должен содержать neurology или neurosurgery"
        )
    return tuple(sorted(tags))


def _public_https_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise LifeOSImportError("source_url некорректен") from exc
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
        raise LifeOSImportError("source_url должен быть публичным HTTPS URL")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address and not address.is_global:
        raise LifeOSImportError("source_url не может указывать на локальный адрес")
    return value


def validate_note(path: Path, vault_root: Path) -> ApprovedNote:
    try:
        size = path.stat().st_size
        raw_bytes = path.read_bytes()
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LifeOSImportError("заметка должна быть в UTF-8") from exc
    if size > MAX_NOTE_BYTES or len(raw) > MAX_NOTE_CHARACTERS:
        raise LifeOSImportError("заметка превышает установленный лимит")
    meta, body = _split_frontmatter(raw)
    if not _truthy(meta.get("vitalis_ingest", "")):
        raise LifeOSImportError("vitalis_ingest не разрешён")
    if meta.get("review_status", "").strip().casefold() != "approved":
        raise LifeOSImportError("review_status должен быть approved")
    if meta.get("content_scope", "").strip() != CONTENT_SCOPE:
        raise LifeOSImportError(f"content_scope должен быть {CONTENT_SCOPE}")

    source_id = _required(meta, "source_id").casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,80}", source_id):
        raise LifeOSImportError("source_id должен быть стабильным ASCII-идентификатором")
    title = _required(meta, "title")
    source_url = _public_https_url(_required(meta, "source_url"))
    source_version = _required(meta, "source_version")
    reviewed_by = _required(meta, "reviewed_by")
    reviewed_at = _required(meta, "reviewed_at")
    try:
        date.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise LifeOSImportError("reviewed_at должен быть датой ISO") from exc
    try:
        year = int(_required(meta, "year"))
    except ValueError as exc:
        raise LifeOSImportError("year должен быть годом") from exc
    if year < 1900 or year > datetime.now().year + 1:
        raise LifeOSImportError("year вне допустимого диапазона")
    jurisdiction = _required(meta, "jurisdiction")
    language = _required(meta, "language").casefold()
    if language not in {"ru", "en"}:
        raise LifeOSImportError("language должен быть ru или en")
    specialty_tags = _specialties(_required(meta, "specialty_tags"))
    if len(body) < 100:
        raise LifeOSImportError("одобренный фрагмент слишком короткий")

    pages = (DocumentPage(number=1, text=body),)
    security = scan_untrusted_text(pages)
    if security.prompt_injection_detected:
        raise LifeOSImportError("обнаружены недоверенные инструкции")
    _, pii = redact_pages(pages)
    if pii.total_replacements or pii.review_required:
        raise LifeOSImportError("обнаружены персональные или страховые идентификаторы")

    return ApprovedNote(
        path=path,
        relative_path=path.relative_to(vault_root).as_posix(),
        source_id=source_id,
        title=title,
        source_url=source_url,
        source_version=source_version,
        year=year,
        jurisdiction=jurisdiction,
        language=language,
        specialty_tags=specialty_tags,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
        body=body,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _frontmatter_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_note(note: ApprovedNote) -> str:
    fields: dict[str, Any] = {
        "title": note.title,
        "year": note.year,
        "source": "Life-OS (проверенная заметка)",
        "source_url": note.source_url,
        "url": note.source_url,
        "source_version": note.source_version,
        "jurisdiction": note.jurisdiction,
        "language": note.language,
        "document_type": "reviewed_lifeos_note",
        "specialty_tags": " ".join(note.specialty_tags),
        "content_scope": CONTENT_SCOPE,
        "review_status": "approved",
        "reviewed_by": note.reviewed_by,
        "reviewed_at": note.reviewed_at,
        "lifeos_source_id": note.source_id,
        "lifeos_origin_path": note.relative_path,
        "lifeos_source_sha256": note.sha256,
        "full_text_checked": False,
    }
    frontmatter = "\n".join(
        f"{key}: {_frontmatter_value(value)}" for key, value in fields.items()
    )
    return (
        f"---\n{frontmatter}\n---\n\n"
        f"# {note.title}\n\n"
        "> Проверенный человеком фрагмент из Life-OS. Он не заменяет официальный "
        "полный текст клинической рекомендации и не может иметь приоритет над ним.\n\n"
        f"{note.body.rstrip()}\n"
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def sync_lifeos_vault(lifeos_vault: Path, vitalis_vault: Path) -> dict[str, Any]:
    if not lifeos_vault.is_dir():
        raise LifeOSImportError("Каталог Life-OS vault не найден")
    target = vitalis_vault / "generated" / "lifeos"
    candidates: list[ApprovedNote] = []
    rejected: list[dict[str, str]] = []
    files = sorted(lifeos_vault.rglob("*.md"))
    for path in files:
        try:
            candidates.append(validate_note(path, lifeos_vault))
        except (OSError, LifeOSImportError) as exc:
            rejected.append(
                {"path": path.relative_to(lifeos_vault).as_posix(), "reason": str(exc)}
            )

    counts: dict[str, int] = {}
    for note in candidates:
        counts[note.source_id] = counts.get(note.source_id, 0) + 1
    approved = [note for note in candidates if counts[note.source_id] == 1]
    for note in candidates:
        if counts[note.source_id] > 1:
            rejected.append(
                {"path": note.relative_path, "reason": "дублирующийся source_id"}
            )

    expected: set[str] = set()
    records: list[dict[str, Any]] = []
    for note in approved:
        filename = f"{GENERATED_PREFIX}{note.source_id}.md"
        expected.add(filename)
        _atomic_text(target / filename, render_note(note))
        records.append(
            {
                "source_id": note.source_id,
                "title": note.title,
                "source_url": note.source_url,
                "source_version": note.source_version,
                "origin_path": note.relative_path,
                "source_sha256": note.sha256,
                "output": filename,
                "reviewed_by": note.reviewed_by,
                "reviewed_at": note.reviewed_at,
                "content_scope": CONTENT_SCOPE,
            }
        )

    removed = 0
    for path in target.glob(f"{GENERATED_PREFIX}*.md"):
        if path.name not in expected:
            path.unlink()
            removed += 1
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "automatic_ingestion": False,
        "review_required": True,
        "source_vault": str(lifeos_vault),
        "scanned": len(files),
        "imported": len(records),
        "rejected": sorted(rejected, key=lambda row: row["path"]),
        "records": sorted(records, key=lambda row: row["source_id"]),
    }
    _atomic_text(target / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {
        "scanned": len(files),
        "imported": len(records),
        "rejected": len(rejected),
        "removed": removed,
        "manifest": str(target / "manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Офлайн-импорт одобренных медицинских заметок Life-OS в Vitalis"
    )
    parser.add_argument("lifeos_vault", type=Path)
    parser.add_argument("vitalis_vault", type=Path)
    args = parser.parse_args()
    stats = sync_lifeos_vault(args.lifeos_vault, args.vitalis_vault)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
