from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.documents.analyzer import DocumentAnalyzer
from app.documents.middleware import DocumentPrivacyMiddleware
from app.documents.session_store import DocumentSessionStore
from app.rag.vault import VaultRAG
from app.routes.documents import create_documents_router
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ModeRequest,
    ModeResponse,
    RebuildResponse,
)
from app.services.ask import AskService

settings = get_settings()
rag = VaultRAG(settings.vault_dir, settings.index_dir)
chunks = rag.load_or_build()
ask_service = AskService(settings, rag)
document_analyzer = DocumentAnalyzer(settings)
document_store = DocumentSessionStore(
    ttl_seconds=settings.document_session_ttl_seconds,
    max_sessions=settings.document_max_sessions,
)


@asynccontextmanager
async def lifespan(_app):
    async def cleanup():
        while True:
            await asyncio.sleep(30)
            document_store.count()  # Remove expired contents even when no user returns.

    task = asyncio.create_task(cleanup())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Vitalis AI API",
    version="1.0.0",
    description="RAG + веб-поиск + DeepSeek для российского и международного контуров; GigaChat — резервная заглушка",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(create_documents_router(settings, document_analyzer, document_store))
app.add_middleware(DocumentPrivacyMiddleware, settings=settings)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        service="vitalis-python",
        llm_mode=settings.llm_mode,
        gigachat=settings.gigachat_ready,
        deepseek=settings.deepseek_ready,
        keys_ready=settings.deepseek_ready,
        vault_chunks=len(rag.chunks),
        web_search=settings.web_search_provider,
        documents_enabled=settings.document_analysis_enabled,
        document_external_llm=settings.document_external_llm_ready,
        document_sessions=document_store.count(),
        insurance_official_search=settings.insurance_official_search_enabled,
    )


@app.post("/mode", response_model=ModeResponse)
def set_mode(body: ModeRequest) -> ModeResponse:
    if body.llm_mode == "live" and not settings.deepseek_ready:
        raise HTTPException(
            status_code=409,
            detail="Live-режим недоступен: ключ DeepSeek не настроен.",
        )
    settings.llm_mode = body.llm_mode
    return ModeResponse(ok=True, llm_mode=settings.llm_mode)


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    try:
        data = await ask_service.ask(
            question=body.question,
            mode=body.mode,
            use_web=body.use_web,
            mkb=body.mkb,
        )
        return AskResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail="Медицинский поиск или генерация временно недоступны.",
        ) from e


@app.post("/rag/rebuild", response_model=RebuildResponse)
def rebuild() -> RebuildResponse:
    n = rag.build()
    return RebuildResponse(ok=True, chunks=n, vault=str(settings.vault_dir))


@app.get("/")
def root() -> dict:
    return {
        "service": "Vitalis AI",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask",
        "documents": "POST /documents/analyze",
    }
