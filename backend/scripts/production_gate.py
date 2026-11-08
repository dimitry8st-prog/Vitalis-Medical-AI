from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from app.rag.source_registry import load_registry, validate_registry


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"{label}: не удалось прочитать {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}: ожидается JSON-объект")
        return {}
    return value


def _validate_fulltexts(vault: Path, expected: int, errors: list[str]) -> None:
    root = vault / "generated" / "guidelines" / "rf"
    manifest = _load_json(root / "manifest.json", "КР РФ", errors)
    if not manifest:
        return
    records = manifest.get("records")
    if manifest.get("sync_status") != "complete" or manifest.get("failures"):
        errors.append("КР РФ: последняя синхронизация не завершена без ошибок")
    if not isinstance(records, list) or len(records) != expected:
        errors.append(f"КР РФ: требуется {expected} проверенных документов")
        return
    identifiers: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            errors.append("КР РФ: запись manifest должна быть объектом")
            continue
        external_id = str(record.get("external_id") or "")
        if not re.fullmatch(r"\d+_\d+", external_id) or external_id in identifiers:
            errors.append(f"КР РФ: некорректный или повторный id {external_id!r}")
        identifiers.add(external_id)
        if record.get("content_scope") != "official_full_text_extracted":
            errors.append(f"КР РФ {external_id}: неверный content_scope")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or "")):
            errors.append(f"КР РФ {external_id}: отсутствует SHA-256")
        try:
            page_count = int(record.get("page_count") or 0)
            character_count = int(record.get("character_count") or 0)
        except (TypeError, ValueError):
            page_count = 0
            character_count = 0
        if page_count < 2:
            errors.append(f"КР РФ {external_id}: недостаточно страниц")
        if character_count < 1_000:
            errors.append(f"КР РФ {external_id}: недостаточно извлечённого текста")
        for field in ("authorized_by", "authorized_at", "source_url", "document_url"):
            if not record.get(field):
                errors.append(f"КР РФ {external_id}: отсутствует {field}")
        expected_output = f"minzdrav-full-{external_id}.md"
        output_value = str(record.get("output") or "")
        if output_value != expected_output:
            errors.append(f"КР РФ {external_id}: некорректное имя output")
        output = root / expected_output
        if not output.is_file():
            errors.append(f"КР РФ {external_id}: отсутствует локальный Markdown")


def _validate_lifeos(path: Path | None, errors: list[str]) -> None:
    """Life-OS may be inventoried, but portal articles must never enter Vitalis RAG."""
    if path is None:
        return
    manifest = _load_json(path, "Life-OS", errors)
    if not manifest:
        return
    if manifest.get("automatic_ingestion") is not False:
        errors.append("Life-OS: automatic_ingestion должен быть false")
    records = manifest.get("records")
    if not isinstance(records, list):
        errors.append("Life-OS: records должен быть массивом")
        return
    if records or int(manifest.get("imported") or 0) != 0:
        errors.append(
            "Life-OS: статьи портала не допускаются в RAG Vitalis; разрешены только "
            "проверенные полные тексты клинических рекомендаций"
        )


def _load_env(path: Path, errors: list[str]) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"Окружение: не удалось прочитать {path}: {exc}")
        return {}
    values: dict[str, str] = {}
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            errors.append(f"Окружение: строка {number} не имеет KEY=VALUE")
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _validate_env(path: Path, errors: list[str]) -> None:
    env = _load_env(path, errors)
    expected = {
        "LLM_MODE": "live",
        "EVIDENCE_GATE_MODE": "strict",
        "DOCUMENT_EXTERNAL_LLM_ENABLED": "false",
        "GIGACHAT_ENABLED": "false",
        "DOCUMENT_REQUIRE_DEIDENTIFIED_CONFIRMATION": "true",
    }
    for key, value in expected.items():
        if env.get(key, "").casefold() != value:
            errors.append(f"Окружение: {key} должен быть {value}")
    if not env.get("DEEPSEEK_API_KEY"):
        errors.append("Окружение: DEEPSEEK_API_KEY не задан")
    if "*" in env.get("CORS_ORIGINS", ""):
        errors.append("Окружение: wildcard CORS запрещён")


def _validate_clinical_report(
    path: Path, variant: str, errors: list[str]
) -> None:
    report = _load_json(path, "Клиническая оценка", errors)
    variants = report.get("variants")
    if not isinstance(variants, dict) or variant not in variants:
        errors.append(f"Клиническая оценка: вариант {variant!r} отсутствует")
        return
    metrics = variants[variant]
    if not isinstance(metrics, dict):
        errors.append(f"Клиническая оценка: вариант {variant!r} повреждён")
        return
    if metrics.get("release_gate_passed") is not True:
        errors.append("Клиническая оценка: release_gate_passed не пройден")


def run_gate(
    *,
    vault: Path,
    env_file: Path,
    clinical_report: Path,
    variant: str,
    expected_fulltexts: int,
    lifeos_manifest: Path | None,
) -> list[str]:
    errors: list[str] = []
    try:
        registry_errors = validate_registry(load_registry(vault / "source_registry.json"))
    except (OSError, ValueError) as exc:
        errors.append(f"Реестр источников: {exc}")
    else:
        errors.extend(f"Реестр источников: {error}" for error in registry_errors)
    _validate_fulltexts(vault, expected_fulltexts, errors)
    _validate_lifeos(lifeos_manifest, errors)
    _validate_env(env_file, errors)
    _validate_clinical_report(clinical_report, variant, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed release gate частного production-пилота Vitalis"
    )
    parser.add_argument("--vault", type=Path, default=Path("vault"))
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--clinical-report", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--expected-fulltexts", type=int, default=17)
    parser.add_argument("--lifeos-manifest", type=Path)
    args = parser.parse_args()
    errors = run_gate(
        vault=args.vault,
        env_file=args.env_file,
        clinical_report=args.clinical_report,
        variant=args.variant,
        expected_fulltexts=args.expected_fulltexts,
        lifeos_manifest=args.lifeos_manifest,
    )
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "release": "private_clinician_pilot",
                "clinical_variant": args.variant,
                "fulltexts": args.expected_fulltexts,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
