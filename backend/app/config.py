from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    python_host: str = "0.0.0.0"
    python_port: int = 8000
    vault_path: str = "../vault"
    # Пусто → %LOCALAPPDATA%/vitalis/faiss_index (FAISS на Windows плохо пишет в пути с кириллицей)
    faiss_index_path: str = ""
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Резервная интеграция: по умолчанию отключена и не участвует в маршрутизации.
    gigachat_enabled: bool = False
    gigachat_client_id: str = ""
    gigachat_client_secret: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    web_search_provider: Literal["duckduckgo", "serper", "tavily"] = "duckduckgo"
    serper_api_key: str = ""
    tavily_api_key: str = ""
    web_fetch_full_text: bool = True
    web_fetch_max_documents: int = 3
    web_fetch_max_bytes: int = 8_000_000

    llm_mode: Literal["mock", "live"] = "mock"
    # warn: отвечать с предупреждением при сбое веб-поиска; strict: блокировать ответ.
    evidence_gate_mode: Literal["warn", "strict"] = "warn"

    # Vitalis Documents. Внешний LLM для документов выключен по умолчанию:
    # включать только после проверки правовой схемы и политики провайдера.
    document_analysis_enabled: bool = True
    document_external_llm_enabled: bool = False
    document_max_file_bytes: int = 8_000_000
    document_max_pages: int = 40
    document_max_chars: int = 80_000
    document_parse_timeout_seconds: int = 20
    document_session_ttl_seconds: int = 1_800
    document_max_sessions: int = 32
    document_require_deidentified_confirmation: bool = True

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def vault_dir(self) -> Path:
        p = Path(self.vault_path)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parents[1] / p).resolve()
        return p

    @property
    def index_dir(self) -> Path:
        raw = (self.faiss_index_path or "").strip()
        if not raw:
            base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".vitalis"))
            p = base / "vitalis" / "faiss_index"
        else:
            p = Path(raw)
            if not p.is_absolute():
                p = (Path(__file__).resolve().parents[1] / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def gigachat_ready(self) -> bool:
        # Сбер: либо Client ID + Secret, либо один Authorization Key в CLIENT_SECRET
        return self.gigachat_enabled and bool(self.gigachat_client_secret.strip())

    @property
    def deepseek_ready(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def document_external_llm_ready(self) -> bool:
        return (
            self.document_analysis_enabled
            and self.document_external_llm_enabled
            and self.llm_mode == "live"
            and self.deepseek_ready
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
