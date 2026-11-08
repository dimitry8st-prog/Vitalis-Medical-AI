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


def _same_pdf_object(left: Any, right: Any) -> bool:
    if left is right or left == right:
        return True
    left_id = getattr(left, "idnum", None)
    right_id = getattr(right, "idnum", None)
    if left_id is not None or right_id is not None:
        return (
            left_id is not None
            and right_id is not None
            and left_id == right_id
            and getattr(left, "generation", None) == getattr(right, "generation", None)
        )
    return _resolve(left) is _resolve(right)


def _is_allowed_c2pa_embedding(root: Any, embedded_files_value: Any) -> bool:
    """Allow only the inert C2PA provenance manifest added to generated PDFs."""

    associated_files = _resolve(root.get("/AF"))
    embedded_files = _resolve(embedded_files_value)
    if not isinstance(associated_files, list) or len(associated_files) != 1:
        return False
    if not hasattr(embedded_files, "get") or embedded_files.get("/Kids") is not None:
        return False

    entries = _resolve(embedded_files.get("/Names"))
    if not isinstance(entries, list) or len(entries) != 2:
        return False
    label, file_spec_ref = entries
    if str(_resolve(label)) != "Content Credentials":
        return False
    if not _same_pdf_object(associated_files[0], file_spec_ref):
        return False

    file_spec = _resolve(file_spec_ref)
    if not hasattr(file_spec, "get"):
        return False
    if file_spec.get("/Type") != "/Filespec":
        return False
    if file_spec.get("/AFRelationship") != "/C2PA_Manifest":
        return False
    if str(file_spec.get("/Subtype", "")).lstrip("/") != "application/c2pa":
        return False
    for key in ("/F", "/UF", "/Desc"):
        if str(file_spec.get(key, "")) != "Content Credentials":
            return False

    embedded_streams = _resolve(file_spec.get("/EF"))
    if not hasattr(embedded_streams, "get"):
        return False
    if not set(embedded_streams).issubset({"/F", "/UF"}) or not embedded_streams:
        return False
    stream_refs = [embedded_streams.get(key) for key in ("/F", "/UF") if embedded_streams.get(key)]
    if not stream_refs or any(not _same_pdf_object(stream_refs[0], ref) for ref in stream_refs[1:]):
        return False

    stream = _resolve(stream_refs[0])
    return bool(
        hasattr(stream, "get")
        and stream.get("/Type") == "/EmbeddedFile"
        and str(stream.get("/Subtype", "")).lstrip("/") == "application/c2pa"
    )


def _contains_catalog_risk(reader: PdfReader) -> bool:
    root = _resolve(reader.trailer.get("/Root"))
    if not hasattr(root, "get"):
        return False

    if root.get("/OpenAction") is not None or root.get("/AA") is not None:
        return True

    names = _resolve(root.get("/Names"))
    embedded_files = names.get("/EmbeddedFiles") if hasattr(names, "get") else None
    if hasattr(names, "get") and names.get("/JavaScript") is not None:
        return True
    if embedded_files is not None:
        if not _is_allowed_c2pa_embedding(root, embedded_files):
            return True
    elif root.get("/AF") is not None:
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
            if (
                annotation.get("/Subtype") in {"/FileAttachment", "/RichMedia"}
                or annotation.get("/AA") is not None
            ):
                return True
            for action_key in ("/A", "/AA"):
                action = _resolve(annotation.get(action_key))
                if hasattr(action, "get") and (
                    action.get("/Next") is not None
                    or action.get("/S")
                    in {
                        "/JavaScript",
                        "/Launch",
                        "/SubmitForm",
                        "/ImportData",
                    }
                ):
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
