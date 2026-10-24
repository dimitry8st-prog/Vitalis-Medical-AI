from __future__ import annotations

import hashlib
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import Settings
from app.documents.analyzer import DocumentAnalysisError, DocumentAnalyzer
from app.documents.extractor import (
    DocumentValidationError,
    read_upload_limited,
    validate_upload_metadata,
)
from app.documents.insurance_search import InsuranceOfficialSearch
from app.documents.isolated_parser import extract_pdf_isolated
from app.documents.models import (
    DocumentDeleteResponse,
    DocumentQuestionRequest,
    DocumentQuestionResponse,
    DocumentSessionResponse,
    InsuranceAnalysis,
    InsuranceWebLookup,
)
from app.documents.redactor import redact_pages
from app.documents.security import scan_untrusted_text
from app.documents.session_store import DocumentSessionStore, StoredDocument

DOCUMENT_DISCLAIMER = (
    "Автоматический разбор предназначен для навигации по документу, не является диагнозом "
    "или назначением и требует проверки врачом по исходному PDF."
)
INSURANCE_DISCLAIMER = (
    "Автоматический разбор помогает ориентироваться в договоре, но не является "
    "юридическим заключением и не гарантирует страховую выплату. Приоритет имеют "
    "ваш полис и указанная в нём редакция правил страхования."
)
logger = logging.getLogger(__name__)


def _safe_session_ref(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]


def _session_response(item: StoredDocument) -> DocumentSessionResponse:
    return DocumentSessionResponse(
        session_id=item.session_id,
        expires_at=item.expires_at,
        document_kind=item.document_kind,
        pages=len(item.pages),
        character_count=item.character_count,
        llm_mode=item.llm_mode,
        analysis=item.analysis,
        insurance_lookup=item.insurance_lookup,
        pii=item.pii,
        security=item.security,
        disclaimer=(
            INSURANCE_DISCLAIMER
            if item.document_kind == "insurance"
            else DOCUMENT_DISCLAIMER
        ),
    )


def create_documents_router(
    settings: Settings,
    analyzer: DocumentAnalyzer,
    store: DocumentSessionStore,
    insurance_search: InsuranceOfficialSearch | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/documents", tags=["documents"])
    official_search = insurance_search or InsuranceOfficialSearch(settings)

    @router.post("/analyze", response_model=DocumentSessionResponse)
    async def analyze_document(
        file: Annotated[UploadFile, File(...)],
        confirmed_deidentified: Annotated[bool, Form()] = False,
        document_kind: Annotated[Literal["medical", "insurance"], Form()] = "medical",
        search_official_sources: Annotated[bool, Form()] = True,
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
            extracted = await extract_pdf_isolated(
                data,
                max_pages=settings.document_max_pages,
                max_chars=settings.document_max_chars,
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
            confirmed_deidentified
            and not pii_report.review_required
            and not security_report.prompt_injection_detected
        )

        try:
            analysis, llm_mode = await analyzer.analyze(
                redacted_pages,
                allow_external=allow_external,
                document_kind=document_kind,
            )
        except DocumentAnalysisError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
            ) from exc

        insurance_lookup: InsuranceWebLookup | None = None
        if document_kind == "insurance":
            if not isinstance(analysis, InsuranceAnalysis):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Получен неверный тип разбора страхового документа.",
                )
            insurance_lookup = await official_search.lookup_missing_coverage(
                redacted_pages,
                analysis,
                requested=search_official_sources,
            )

        item = store.create(
            pages=redacted_pages,
            analysis=analysis,
            pii=pii_report,
            security=security_report,
            llm_mode=llm_mode,
            document_kind=document_kind,
            insurance_lookup=insurance_lookup,
        )
        logger.info(
            "document_analysis_completed session=%s kind=%s pages=%d chars=%d mode=%s pii_replacements=%d injection=%s",
            _safe_session_ref(item.session_id),
            item.document_kind,
            len(item.pages),
            item.character_count,
            item.llm_mode,
            item.pii.total_replacements,
            item.security.prompt_injection_detected,
        )
        return _session_response(item)

    @router.get("/{session_id}", response_model=DocumentSessionResponse)
    async def get_document_session(session_id: str) -> DocumentSessionResponse:
        if not settings.document_analysis_enabled:
            raise HTTPException(status_code=503, detail="Модуль документов отключён.")
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
        if not settings.document_analysis_enabled:
            raise HTTPException(status_code=503, detail="Модуль документов отключён.")
        item = store.get(session_id)
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Сессия не найдена или истекла.",
            )
        if (
            item.document_kind == "insurance"
            and body.use_official_web
            and isinstance(item.analysis, InsuranceAnalysis)
            and (
                item.insurance_lookup is None
                or item.insurance_lookup.status == "not_requested"
            )
        ):
            lookup = await official_search.lookup_missing_coverage(
                item.pages,
                item.analysis,
                requested=True,
            )
            item = store.set_insurance_lookup(session_id, lookup) or item

        sources = (
            item.insurance_lookup.sources
            if body.use_official_web and item.insurance_lookup is not None
            else []
        )
        try:
            answer, llm_mode = await analyzer.answer_question(
                item.pages,
                body.question,
                allow_external=item.llm_mode == "live",
                document_kind=item.document_kind,
                official_sources=sources,
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
            document_kind=item.document_kind,
            llm_mode=llm_mode,
            answer=answer.answer,
            evidence=answer.evidence,
            official_sources=answer.official_sources,
            limitations=answer.limitations,
            disclaimer=(
                INSURANCE_DISCLAIMER
                if item.document_kind == "insurance"
                else DOCUMENT_DISCLAIMER
            ),
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
