from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    mode: Literal["rf", "intl", "both"] = "both"
    use_web: bool = True
    mkb: str | None = None


class AskResponse(BaseModel):
    question: str
    mode: str
    llm_mode: str
    card: dict[str, Any]
    rf: dict[str, Any] | None = None
    intl: dict[str, Any] | None = None
    rag: list[dict[str, Any]] = []
    web: list[dict[str, Any]] = []
    search: dict[str, Any] | None = None
    learning: dict[str, Any] | None = None
    disclaimer: str = "Информация носит справочный характер. Решение принимает лечащий врач."


class RebuildResponse(BaseModel):
    ok: bool
    chunks: int
    vault: str


class HealthResponse(BaseModel):
    ok: bool
    service: str
    llm_mode: str
    gigachat: bool
    deepseek: bool
    keys_ready: bool
    vault_chunks: int
    web_search: str
    documents_enabled: bool = False
    document_external_llm: bool = False
    document_sessions: int = 0


class ModeRequest(BaseModel):
    llm_mode: Literal["mock", "live"]


class ModeResponse(BaseModel):
    ok: bool
    llm_mode: Literal["mock", "live"]
