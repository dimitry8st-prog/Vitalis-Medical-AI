from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from app.documents.models import (
    DocumentAnalysis,
    DocumentPage,
    PiiReport,
    SecurityReport,
)


@dataclass(frozen=True, slots=True)
class StoredDocument:
    session_id: str
    created_at_epoch: float
    expires_at_epoch: float
    pages: tuple[DocumentPage, ...]
    analysis: DocumentAnalysis
    pii: PiiReport
    security: SecurityReport
    llm_mode: Literal["mock", "live", "disabled"]

    @property
    def expires_at(self) -> datetime:
        return datetime.fromtimestamp(self.expires_at_epoch, tz=UTC)

    @property
    def character_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


class DocumentSessionStore:
    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_sessions: int,
        clock: Callable[[], float] = time.time,
    ):
        self.ttl_seconds = max(60, ttl_seconds)
        self.max_sessions = max(1, max_sessions)
        self._clock = clock
        self._sessions: dict[str, StoredDocument] = {}
        self._lock = threading.RLock()

    def _purge_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, item in self._sessions.items()
            if item.expires_at_epoch <= now
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def create(
        self,
        *,
        pages: tuple[DocumentPage, ...],
        analysis: DocumentAnalysis,
        pii: PiiReport,
        security: SecurityReport,
        llm_mode: Literal["mock", "live", "disabled"],
    ) -> StoredDocument:
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            while len(self._sessions) >= self.max_sessions:
                oldest_id = min(
                    self._sessions,
                    key=lambda session_id: self._sessions[session_id].created_at_epoch,
                )
                del self._sessions[oldest_id]
            session_id = secrets.token_urlsafe(24)
            item = StoredDocument(
                session_id=session_id,
                created_at_epoch=now,
                expires_at_epoch=now + self.ttl_seconds,
                pages=pages,
                analysis=analysis,
                pii=pii,
                security=security,
                llm_mode=llm_mode,
            )
            self._sessions[session_id] = item
            return item

    def get(self, session_id: str) -> StoredDocument | None:
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def count(self) -> int:
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            return len(self._sessions)
