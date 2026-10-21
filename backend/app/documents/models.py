from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class DocumentPage:
    number: int
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    pages: tuple[DocumentPage, ...]
    sha256: str
    byte_size: int
    character_count: int


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceItem(StrictModel):
    category: Literal[
        "diagnosis",
        "measurement",
        "medication",
        "finding",
        "contradiction",
        "red_flag",
        "other",
    ] = "other"
    text: str = Field(..., min_length=1, max_length=2_000)
    page: int = Field(..., ge=1)
    source_quote: str = Field(..., min_length=1, max_length=500)


class DocumentAnalysis(StrictModel):
    document_type: str = Field(default="Не определён", max_length=200)
    summary: str = Field(..., min_length=1, max_length=4_000)
    diagnoses: list[EvidenceItem] = Field(default_factory=list)
    measurements: list[EvidenceItem] = Field(default_factory=list)
    medications: list[EvidenceItem] = Field(default_factory=list)
    findings: list[EvidenceItem] = Field(default_factory=list)
    contradictions: list[EvidenceItem] = Field(default_factory=list)
    red_flags: list[EvidenceItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    questions_for_doctor: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class PiiReport(StrictModel):
    replacement_counts: dict[str, int] = Field(default_factory=dict)
    total_replacements: int = 0
    review_required: bool = False
    warnings: list[str] = Field(default_factory=list)


class SecurityWarning(StrictModel):
    kind: str
    page: int | None = None


class SecurityReport(StrictModel):
    prompt_injection_detected: bool = False
    warnings: list[SecurityWarning] = Field(default_factory=list)


class DocumentSessionResponse(StrictModel):
    session_id: str
    expires_at: datetime
    pages: int
    character_count: int
    llm_mode: Literal["mock", "live", "disabled"]
    analysis: DocumentAnalysis
    pii: PiiReport
    security: SecurityReport
    disclaimer: str


class DocumentQuestionRequest(StrictModel):
    question: str = Field(..., min_length=3, max_length=1_000)


class DocumentQuestionResponse(StrictModel):
    session_id: str
    llm_mode: Literal["mock", "live", "disabled"]
    answer: str = Field(..., min_length=1, max_length=6_000)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str


class DocumentDeleteResponse(StrictModel):
    ok: bool
    deleted: bool
