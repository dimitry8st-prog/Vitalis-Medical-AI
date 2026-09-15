from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.rag.vault import VaultRAG
from app.schemas import AskRequest, AskResponse, HealthResponse, RebuildResponse
from app.services.ask import AskService

settings = get_settings()
rag = VaultRAG(settings.vault_dir, settings.index_dir)
chunks = rag.load_or_build()
ask_service = AskService(settings, rag)

app = FastAPI(
    title="Vitalis AI API",
    version="1.0.0",
    description="RAG + веб-поиск + DeepSeek для российского и международного контуров; GigaChat — резервная заглушка",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    )


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
        raise HTTPException(status_code=500, detail=str(e)) from e


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
    }
