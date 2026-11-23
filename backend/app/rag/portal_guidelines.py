from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

TARGET_SPECIALTIES = {
    "neurology": "Неврология",
    "neurosurgery": "Нейрохирургия",
}
SOURCE_ID_PREFIX = "ru-minzdrav-neuro-"
GENERATED_FILE_PREFIX = "minzdrav-"
CONTENT_SCOPE = "bibliographic_metadata_only"


class PortalGuidelineError(ValueError):
    pass


def _official_minzdrav_url(url: str, external_id: str) -> str:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError as exc:
        raise PortalGuidelineError(f"{external_id}: некорректный URL") from exc
    expected_path = f"/view-cr/{external_id}"
    if (
        parsed.scheme != "https"
        or host != "cr.minzdrav.gov.ru"
        or parsed.username
        or parsed.password
        or port not in (None, 443)
        or parsed.path.rstrip("/") != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise PortalGuidelineError(
            f"{external_id}: ожидается официальная HTTPS-карточка {expected_path}"
        )
    return f"https://cr.minzdrav.gov.ru{expected_path}"


def _publication_year(value: Any, external_id: str) -> int:
    raw = str(value or "")[:10]
    for parser in (
        date.fromisoformat,
        lambda item: datetime.strptime(item, "%d.%m.%Y").date(),
    ):
        try:
            return parser(raw).year
        except ValueError:
            continue
    raise PortalGuidelineError(f"{external_id}: отсутствует корректная дата публикации")


def _clean_line(value: Any, field: str, external_id: str) -> str:
    clean = re.sub(r"\s+", " ", str(value or "")).strip()
    if not clean:
        raise PortalGuidelineError(f"{external_id}: отсутствует {field}")
    return clean


def _codes(value: Any) -> list[str]:
    return list(
        dict.fromkeys(
            code.upper()
            for code in re.findall(
                r"(?<![A-ZА-Я0-9])[A-ZА-Я]\d{2}(?:\.\d{1,4})?",
                str(value or "").upper(),
            )
        )
    )


def select_portal_guidelines(portal_registry: dict[str, Any]) -> list[dict[str, Any]]:
    if portal_registry.get("schema_version") != 1:
        raise PortalGuidelineError("Реестр портала должен иметь schema_version=1")
    items = portal_registry.get("items")
    if not isinstance(items, list):
        raise PortalGuidelineError("В реестре портала отсутствует массив items")

    selected: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise PortalGuidelineError("Элемент реестра портала должен быть объектом")
        if raw.get("source_key") != "minzdrav" or raw.get("status") != "active":
            continue
        specialty_tags = tuple(
            slug
            for slug in TARGET_SPECIALTIES
            if slug in (raw.get("specialty_tags") or [])
        )
        if not specialty_tags:
            continue
        external_id = _clean_line(raw.get("external_id"), "external_id", "unknown")
        if not re.fullmatch(r"\d+_\d+", external_id):
            raise PortalGuidelineError(f"{external_id}: недопустимый идентификатор Минздрава")
        if external_id in identifiers:
            raise PortalGuidelineError(f"{external_id}: дублирующийся идентификатор")
        identifiers.add(external_id)
        selected.append(
            {
                "external_id": external_id,
                "title": _clean_line(raw.get("title"), "title", external_id),
                "organization": _clean_line(
                    raw.get("organization") or "Минздрав России",
                    "organization",
                    external_id,
                ),
                "source_version": _clean_line(
                    raw.get("version") or external_id, "version", external_id
                ),
                "publication_year": _publication_year(
                    raw.get("published_on"), external_id
                ),
                "published_on": str(raw.get("published_on") or "")[:10],
                "codes": _codes(raw.get("codes")),
                "specialty_tags": list(specialty_tags),
                "url": _official_minzdrav_url(str(raw.get("url") or ""), external_id),
            }
        )
    return sorted(selected, key=lambda row: (row["title"].casefold(), row["external_id"]))


def _frontmatter_value(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def render_rag_document(item: dict[str, Any]) -> str:
    codes = ", ".join(item["codes"])
    specialties = ", ".join(
        TARGET_SPECIALTIES[slug] for slug in item["specialty_tags"]
    )
    primary_code = item["codes"][0] if item["codes"] else ""
    fields = {
        "title": item["title"],
        "mkb": primary_code,
        "mkb_codes": codes,
        "year": item["publication_year"],
        "published_on": item["published_on"],
        "source": "Минздрав России",
        "organization": item["organization"],
        "jurisdiction": "RU",
        "language": "ru",
        "document_type": "clinical_guideline_metadata",
        "source_version": item["source_version"],
        "source_record_id": item["external_id"],
        "guideline_status": "active",
        "current_version_checked_at": item.get("current_version_checked_at", ""),
        "url": item["url"],
        "source_url": item["url"],
        "specialty_tags": " ".join(item["specialty_tags"]),
        "content_scope": CONTENT_SCOPE,
        "review_status": "approved_metadata_only",
    }
    frontmatter = "\n".join(
        f"{key}: {_frontmatter_value(value)}" for key, value in fields.items()
    )
    return (
        f"---\n{frontmatter}\n---\n\n"
        f"# {item['title']}\n\n"
        "> В RAG загружена только официальная библиографическая карточка. "
        "Она помогает найти нужную КР, но не является источником утверждений "
        "о диагностике, лечении, дозировках или классе доказательности.\n\n"
        "## Профили\n\n"
        f"{specialties}.\n\n"
        "## Коды МКБ\n\n"
        f"{codes or 'Не указаны в карточке портала'}.\n\n"
        "## Официальная карточка Минздрава\n\n"
        f"[{item['url']}]({item['url']})\n\n"
        "Для медицинского ответа необходимо проверить действующий полный текст "
        "по официальной ссылке и использовать извлечённый фрагмент первоисточника.\n"
    )


def _source_record(
    item: dict[str, Any], *, reviewed_at: str, reviewed_by: str
) -> dict[str, Any]:
    topics = list(item["specialty_tags"])
    topics.extend(f"mkb:{code}" for code in item["codes"][:12])
    return {
        "id": f"{SOURCE_ID_PREFIX}{item['external_id']}",
        "title": item["title"],
        "url": item["url"],
        "organization": item["organization"],
        "jurisdiction": "RU",
        "language": "ru",
        "document_type": "clinical_guideline_metadata",
        "content_scope": CONTENT_SCOPE,
        "publication_year": item["publication_year"],
        "source_version": item["source_version"],
        "guideline_status": "active",
        "current_version_checked_at": reviewed_at,
        "revision_due": None,
        "checked_at": reviewed_at,
        "status": "approved",
        "ingest_allowed": True,
        "topics": topics,
        "verification_level": "official_registry_metadata",
        "review_notes": (
            "Одобрена загрузка только библиографических метаданных официальной "
            "карточки Минздрава. Полный текст и медицинские утверждения не "
            "рассматривались и требуют отдельной врачебной проверки."
        ),
        "reviewed_at": reviewed_at,
        "reviewed_by": reviewed_by,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sync_portal_guidelines(
    portal_registry_path: Path,
    vault_dir: Path,
    source_registry_path: Path,
    *,
    reviewed_at: str,
    reviewed_by: str,
) -> dict[str, int]:
    try:
        date.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise PortalGuidelineError("reviewed_at должен быть датой ISO") from exc
    if not reviewed_by.strip():
        raise PortalGuidelineError("reviewed_by не может быть пустым")

    portal_registry = json.loads(portal_registry_path.read_text(encoding="utf-8"))
    selected = select_portal_guidelines(portal_registry)
    target_dir = vault_dir / "guidelines" / "rf"
    target_dir.mkdir(parents=True, exist_ok=True)

    expected_files: set[str] = set()
    for item in selected:
        item = {**item, "current_version_checked_at": reviewed_at}
        name = f"{GENERATED_FILE_PREFIX}{item['external_id']}.md"
        expected_files.add(name)
        (target_dir / name).write_text(render_rag_document(item), encoding="utf-8")

    removed = 0
    for path in target_dir.glob(f"{GENERATED_FILE_PREFIX}*.md"):
        if path.name not in expected_files:
            path.unlink()
            removed += 1

    source_registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    preserved = [
        source
        for source in source_registry.get("sources", [])
        if not str(source.get("id") or "").startswith(SOURCE_ID_PREFIX)
    ]
    generated = [
        _source_record(item, reviewed_at=reviewed_at, reviewed_by=reviewed_by)
        for item in selected
    ]
    source_registry["updated_at"] = reviewed_at
    source_registry["sources"] = preserved + generated
    _atomic_json(source_registry_path, source_registry)
    return {
        "selected": len(selected),
        "written": len(expected_files),
        "removed": removed,
        "source_records": len(generated),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Импорт официальных метаданных КР Минздрава из реестра портала"
    )
    parser.add_argument("portal_registry", type=Path)
    parser.add_argument("vault_dir", type=Path)
    parser.add_argument("source_registry", type=Path)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--reviewed-by", required=True)
    args = parser.parse_args()
    stats = sync_portal_guidelines(
        args.portal_registry,
        args.vault_dir,
        args.source_registry,
        reviewed_at=args.reviewed_at,
        reviewed_by=args.reviewed_by,
    )
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
