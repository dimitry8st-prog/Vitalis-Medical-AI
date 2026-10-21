from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Annotated

import anyio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import Settings
from app.documents.analyzer import DocumentAnalysisError, DocumentAnalyzer
from app.documents.extractor import (
    DocumentValidationError,
    extract_pdf,
    read_upload_limited,
    validate_upload_metadata,
)
from app.documents.models import (
    DocumentDeleteResponse,
    DocumentQuestionRequest,
    DocumentQuestionResponse,
    DocumentSessionResponse,
)
from app.documents.redactor import redact_pages
from app.documents.security import scan_untrusted_text
from app.documents.session_store import DocumentSessionStore, StoredDocument

DOCUMENT_DISCLAIMER = (
    "Автоматический разбор предназначен для навигации по документу, не является диагнозом "
    "или назначением и требует проверки врачом по исходному PDF."
)
logger = logging.getLogger(__name__)


def _safe_session_ref(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]


def _session_response(item: StoredDocument) -> DocumentSessionResponse:
    return DocumentSessionResponse(
        session_id=item.session_id,
        expires_at=item.expires_at,
        pages=len(item.pages),
        character_count=item.character_count,
        llm_mode=item.llm_mode,
        analysis=item.analysis,
        pii=item.pii,
        security=item.security,
        disclaimer=DOCUMENT_DISCLAIMER,
    )


def create_documents_router(
    settings: Settings,
    analyzer: DocumentAnalyzer,
    store: DocumentSessionStore,
) -> APIRouter:
    router = APIRouter(prefix="/documents", tags=["documents"])

    @router.post("/analyze", response_model=DocumentSessionResponse)
    async def analyze_document(
        file: Annotated[UploadFile, File(...)],
        confirmed_deidentified: Annotated[bool, Form()] = False,
    ) -> DocumentSessionResponse:
        if not settings.document_analysis_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Модуль документов отключён.",
            )
        if (
            settings.document_require_deidentified_confirmation
            and not confirmed_deidentified
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Подтвердите, что PDF синтетический или заранее обезличенный.",
            )

        try:
            validate_upload_metadata(file.filename, file.content_type)
            data = await read_upload_limited(file, settings.document_max_file_bytes)
            extracted = await asyncio.wait_for(
                anyio.to_thread.run_sync(
                    lambda: extract_pdf(
                        data,
                        max_pages=settings.document_max_pages,
                        max_chars=settings.document_max_chars,
                    ),
                    abandon_on_cancel=True,
                ),
                timeout=settings.document_parse_timeout_seconds,
            )
        except DocumentValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Проверка PDF превысила допустимое время.",
            ) from exc
        finally:
            await file.close()

        data = b""
        redacted_pages, pii_report = redact_pages(extracted.pages)
        security_report = scan_untrusted_text(redacted_pages)
        allow_external = (
            not pii_report.review_required
            and not security_report.prompt_injection_detected
        )

        try:
            analysis, llm_mode = await analyzer.analyze(
                redacted_pages,
                allow_external=allow_external,
            )
        except DocumentAnalysisError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
            ) from exc

        item = store.create(
            pages=redacted_pages,
            analysis=analysis,
            pii=pii_report,
            security=security_report,
            llm_mode=llm_mode,
        )
        logger.info(
            "document_analysis_completed session=%s pages=%d chars=%d mode=%s pii_replacements=%d injection=%s",
            _safe_session_ref(item.session_id),
            len(item.pages),
            item.character_count,
            item.llm_mode,
            item.pii.total_replacements,
            item.security.prompt_injection_detected,
        )
        return _session_response(item)

    @router.get("/{session_id}", response_model=DocumentSessionResponse)
    async def get_document_session(session_id: str) -> DocumentSessionResponse:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Сессия не найдена или истекла.",
            )
        return _session_response(item)

    @router.post("/{session_id}/questions", response_model=DocumentQuestionResponse)
    async def ask_document_question(
        session_id: str,
        body: DocumentQuestionRequest,
    ) -> DocumentQuestionResponse:
        item = store.get(session_id)
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Сессия не найдена или истекла.",
            )
        try:
            answer, llm_mode = await analyzer.answer_question(
                item.pages,
                body.question,
                allow_external=item.llm_mode == "live",
            )
        except DocumentAnalysisError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
            ) from exc
        logger.info(
            "document_question_completed session=%s mode=%s evidence=%d",
            _safe_session_ref(session_id),
            llm_mode,
            len(answer.evidence),
        )
        return DocumentQuestionResponse(
            session_id=session_id,
            llm_mode=llm_mode,
            answer=answer.answer,
            evidence=answer.evidence,
            limitations=answer.limitations,
            disclaimer=DOCUMENT_DISCLAIMER,
        )

    @router.delete("/{session_id}", response_model=DocumentDeleteResponse)
    async def delete_document_session(session_id: str) -> DocumentDeleteResponse:
        deleted = store.delete(session_id)
        logger.info(
            "document_session_delete session=%s deleted=%s",
            _safe_session_ref(session_id),
            deleted,
        )
        return DocumentDeleteResponse(ok=True, deleted=deleted)

    return router
