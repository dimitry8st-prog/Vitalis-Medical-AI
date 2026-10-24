from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias

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
        "insurance_type",
        "coverage",
        "exclusion",
        "limit",
        "deductible",
        "validity",
        "claim_procedure",
        "contact",
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


class InsuranceAnalysis(StrictModel):
    document_type: str = Field(default="Страховой документ", max_length=200)
    summary: str = Field(..., min_length=1, max_length=4_000)
    insurer_name: str | None = Field(default=None, max_length=200)
    insurer_domain: str | None = Field(default=None, max_length=253)
    insurance_type: str = Field(default="Не определён", max_length=200)
    policy_name: str | None = Field(default=None, max_length=300)
    coverage: list[EvidenceItem] = Field(default_factory=list)
    exclusions: list[EvidenceItem] = Field(default_factory=list)
    limits: list[EvidenceItem] = Field(default_factory=list)
    deductibles: list[EvidenceItem] = Field(default_factory=list)
    validity: list[EvidenceItem] = Field(default_factory=list)
    claim_procedure: list[EvidenceItem] = Field(default_factory=list)
    contacts: list[EvidenceItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    questions_for_insurer: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


AnalysisPayload: TypeAlias = DocumentAnalysis | InsuranceAnalysis


class OfficialSourceEvidence(StrictModel):
    title: str = Field(..., min_length=1, max_length=500)
    url: str = Field(..., min_length=8, max_length=2_000)
    domain: str = Field(..., min_length=3, max_length=253)
    snippet: str = Field(default="", max_length=2_000)
    evidence_excerpt: str = Field(default="", max_length=6_000)
    retrieved_at: datetime | None = None


class InsuranceWebLookup(StrictModel):
    status: Literal[
        "not_requested",
        "not_needed",
        "found",
        "not_found",
        "unsupported_insurer",
        "blocked",
    ]
    insurer_name: str | None = Field(default=None, max_length=200)
    insurer_domain: str | None = Field(default=None, max_length=253)
    query: str | None = Field(default=None, max_length=500)
    sources: list[OfficialSourceEvidence] = Field(default_factory=list)
    warning: str | None = Field(default=None, max_length=1_000)


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
    document_kind: Literal["medical", "insurance"] = "medical"
    pages: int
    character_count: int
    llm_mode: Literal["mock", "live", "disabled"]
    analysis: AnalysisPayload
    insurance_lookup: InsuranceWebLookup | None = None
    pii: PiiReport
    security: SecurityReport
    disclaimer: str


class DocumentQuestionRequest(StrictModel):
    question: str = Field(..., min_length=3, max_length=1_000)
    use_official_web: bool = True


class DocumentQuestionResponse(StrictModel):
    session_id: str
    document_kind: Literal["medical", "insurance"] = "medical"
    llm_mode: Literal["mock", "live", "disabled"]
    answer: str = Field(..., min_length=1, max_length=6_000)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    official_sources: list[OfficialSourceEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str


class DocumentDeleteResponse(StrictModel):
    ok: bool
    deleted: bool
