from __future__ import annotations

import hashlib
import re
from io import BytesIO
from typing import Any

from fastapi import UploadFile
from pypdf import PdfReader

from app.documents.models import DocumentPage, ExtractedDocument


class DocumentValidationError(ValueError):
    """A safe, user-facing validation error without document contents."""


_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def validate_upload_metadata(filename: str | None, content_type: str | None) -> None:
    if not filename or not filename.lower().endswith(".pdf"):
        raise DocumentValidationError("Разрешены только файлы PDF.")
    if content_type and content_type.lower() not in _ALLOWED_CONTENT_TYPES:
        raise DocumentValidationError("Тип файла не соответствует PDF.")


async def read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    data = bytearray()
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise DocumentValidationError(
                f"PDF превышает допустимый размер {max_bytes // 1_000_000} МБ."
            )
    if not data:
        raise DocumentValidationError("Получен пустой файл.")
    return bytes(data)


def _resolve(value: Any) -> Any:
    getter = getattr(value, "get_object", None)
    return getter() if callable(getter) else value


def _contains_catalog_risk(reader: PdfReader) -> bool:
    root = _resolve(reader.trailer.get("/Root"))
    if not hasattr(root, "get"):
        return False

    if root.get("/OpenAction") is not None or root.get("/AA") is not None:
        return True

    names = _resolve(root.get("/Names"))
    if hasattr(names, "get") and (
        names.get("/JavaScript") is not None or names.get("/EmbeddedFiles") is not None
    ):
        return True

    acroform = _resolve(root.get("/AcroForm"))
    if hasattr(acroform, "get") and acroform.get("/XFA") is not None:
        return True

    for page in reader.pages:
        if page.get("/AA") is not None:
            return True
        annotations = _resolve(page.get("/Annots")) or []
        for annotation_ref in annotations:
            annotation = _resolve(annotation_ref)
            if not hasattr(annotation, "get"):
                continue
            for action_key in ("/A", "/AA"):
                action = _resolve(annotation.get(action_key))
                if hasattr(action, "get") and action.get("/S") in {
                    "/JavaScript",
                    "/Launch",
                    "/SubmitForm",
                    "/ImportData",
                }:
                    return True
    return False


def _normalise_text(value: str) -> str:
    value = _CONTROL_CHARS.sub(" ", value or "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def extract_pdf(
    data: bytes,
    *,
    max_pages: int,
    max_chars: int,
) -> ExtractedDocument:
    if not data.startswith(b"%PDF-"):
        raise DocumentValidationError("Сигнатура файла не соответствует PDF.")

    try:
        reader = PdfReader(BytesIO(data), strict=True)
    except Exception as exc:
        raise DocumentValidationError(
            "PDF повреждён или имеет неподдерживаемую структуру."
        ) from exc

    if reader.is_encrypted:
        raise DocumentValidationError("Защищённые паролем PDF не поддерживаются.")
    if not reader.pages:
        raise DocumentValidationError("В PDF нет страниц.")
    if len(reader.pages) > max_pages:
        raise DocumentValidationError(f"В PDF больше допустимых {max_pages} страниц.")
    if _contains_catalog_risk(reader):
        raise DocumentValidationError(
            "PDF содержит активное содержимое или вложения и отклонён из соображений безопасности."
        )

    pages: list[DocumentPage] = []
    total = 0
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            text = _normalise_text(page.extract_text() or "")
            total += len(text)
            if total > max_chars:
                raise DocumentValidationError(
                    f"Извлечённый текст превышает допустимые {max_chars} символов."
                )
            pages.append(DocumentPage(number=page_number, text=text))
    except DocumentValidationError:
        raise
    except Exception as exc:
        raise DocumentValidationError(
            "Не удалось безопасно извлечь текст из PDF."
        ) from exc

    if total < 20:
        raise DocumentValidationError(
            "В PDF почти нет текстового слоя. Для скана сначала нужен OCR."
        )

    return ExtractedDocument(
        pages=tuple(pages),
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        character_count=total,
    )
