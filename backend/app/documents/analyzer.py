from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from app.documents.models import DocumentAnalysis, DocumentPage, EvidenceItem
from app.documents.prompts import ANALYSIS_SYSTEM_PROMPT, QUESTION_SYSTEM_PROMPT
from app.llm.deepseek import DeepSeekClient


class DocumentAnalysisError(RuntimeError):
    """A model response could not pass the document-analysis contract."""


class CompletionClient(Protocol):
    @property
    def ready(self) -> bool: ...

    async def complete(self, system: str, user: str) -> dict[str, Any]: ...


class QuestionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1, max_length=6_000)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


_DIRECT_MEDICAL_ACTION = re.compile(
    r"(?i)\b(?:назначь|назначить|отменить|начать\s+принимать|что\s+принимать|как\s+лечить|"
    r"подбери\s+лечени|измени\s+доз|дозировк[ау]\s+для\s+меня)"
)
_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{4,}")
_LABELLED_LINE = re.compile(
    r"(?i)^(?:диагноз|заключение|результат|рекомендации?|назначения?)\s*[:—-].+"
)
_MEASUREMENT = re.compile(
    r"(?i)\b\d+(?:[.,]\d+)?\s*(?:ммоль/л|мкмоль/л|г/л|мг/л|ед/л|мм\s*рт\.?\s*ст\.?)\b"
)


def _json_from_content(content: str) -> dict[str, Any]:
    value = (content or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise DocumentAnalysisError(
                "Модель вернула ответ вне JSON-контракта."
            ) from exc
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError as nested_exc:
            raise DocumentAnalysisError(
                "Модель вернула повреждённый JSON."
            ) from nested_exc
    if not isinstance(parsed, dict):
        raise DocumentAnalysisError("Модель вернула неверный тип JSON.")
    return parsed


def _normalise_quote(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _verified_items(
    items: list[EvidenceItem], pages: tuple[DocumentPage, ...]
) -> tuple[list[EvidenceItem], int]:
    page_text = {page.number: _normalise_quote(page.text) for page in pages}
    accepted: list[EvidenceItem] = []
    rejected = 0
    for item in items:
        quote = _normalise_quote(item.source_quote)
        if quote and quote in page_text.get(item.page, ""):
            accepted.append(item)
        else:
            rejected += 1
    return accepted, rejected


def verify_analysis_citations(
    analysis: DocumentAnalysis, pages: tuple[DocumentPage, ...]
) -> DocumentAnalysis:
    updates: dict[str, Any] = {}
    rejected = 0
    for field_name in (
        "diagnoses",
        "measurements",
        "medications",
        "findings",
        "contradictions",
        "red_flags",
    ):
        verified, count = _verified_items(getattr(analysis, field_name), pages)
        updates[field_name] = verified
        rejected += count
    summary_evidence = [
        item
        for field_name in ("diagnoses", "measurements", "medications", "findings")
        for item in updates[field_name]
    ][:4]
    if summary_evidence:
        quotes = " ".join(f"«{item.source_quote}»" for item in summary_evidence)
        updates["summary"] = f"Ключевые проверяемые фрагменты документа: {quotes}"
    else:
        updates["summary"] = (
            "Структурированные медицинские пункты с проверяемыми цитатами не получены. "
            "Проверьте исходный PDF вручную."
        )
    updates["limitations"] = [
        *analysis.limitations,
        "Краткая выжимка сформирована только из цитат, найденных на указанных страницах.",
    ]
    if rejected:
        updates["limitations"] = [
            *updates["limitations"],
            f"Исключено пунктов без проверяемой постраничной цитаты: {rejected}.",
        ]
    return analysis.model_copy(update=updates)


def _detect_document_type(pages: tuple[DocumentPage, ...]) -> str:
    text = " ".join(page.text for page in pages).casefold()
    if any(
        key in text
        for key in ("общий анализ крови", "биохимический анализ", "лаборатор")
    ):
        return "Лабораторный результат"
    if any(
        key in text
        for key in ("магнитно-резонанс", "мрт", "компьютерн", "кт ", "рентген")
    ):
        return "Заключение инструментального исследования"
    if "выписн" in text or "эпикриз" in text:
        return "Выписка или эпикриз"
    if "рецепт" in text:
        return "Рецепт"
    return "Медицинский документ"


def _offline_analysis(
    pages: tuple[DocumentPage, ...], mode: Literal["mock", "disabled"]
) -> DocumentAnalysis:
    findings: list[EvidenceItem] = []
    measurements: list[EvidenceItem] = []
    diagnoses: list[EvidenceItem] = []

    for page in pages:
        for line in page.text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if _LABELLED_LINE.search(clean) and len(findings) < 12:
                category = (
                    "diagnosis" if clean.casefold().startswith("диагноз") else "finding"
                )
                item = EvidenceItem(
                    category=category,
                    text=clean[:500],
                    page=page.number,
                    source_quote=clean[:500],
                )
                if category == "diagnosis":
                    diagnoses.append(item)
                else:
                    findings.append(item)
            elif _MEASUREMENT.search(clean) and len(measurements) < 12:
                measurements.append(
                    EvidenceItem(
                        category="measurement",
                        text=clean[:500],
                        page=page.number,
                        source_quote=clean[:500],
                    )
                )

    reason = (
        "Внешний AI-анализ заблокирован проверкой безопасности; показано только "
        "детерминированное извлечение."
        if mode == "disabled"
        else "Внешний AI-анализ выключен; показано только детерминированное извлечение."
    )
    return DocumentAnalysis(
        document_type=_detect_document_type(pages),
        summary=(
            "PDF успешно проверен, текст извлечён по страницам и обезличен. "
            "Ниже отображаются только явно размеченные строки и показатели."
        ),
        diagnoses=diagnoses,
        measurements=measurements,
        findings=findings,
        questions_for_doctor=[
            "Какие сведения из документа требуют клинической проверки?",
            "Нужны ли дополнительные исследования или исходные материалы?",
        ],
        limitations=[
            reason,
            "Сканированные страницы без текстового слоя не распознаются.",
        ],
    )


class DocumentAnalyzer:
    def __init__(self, settings: Settings, client: CompletionClient | None = None):
        self.settings = settings
        self.client = client or DeepSeekClient(settings)

    @property
    def ready(self) -> bool:
        return self.settings.document_external_llm_enabled and self.client.ready

    async def analyze(
        self,
        pages: tuple[DocumentPage, ...],
        *,
        allow_external: bool = True,
    ) -> tuple[DocumentAnalysis, Literal["mock", "live", "disabled"]]:
        if not allow_external:
            return _offline_analysis(pages, "disabled"), "disabled"
        if not self.ready:
            return _offline_analysis(pages, "mock"), "mock"

        user_payload = json.dumps(
            {
                "provenance": "uploaded_medical_pdf_untrusted",
                "pages": [{"page": page.number, "text": page.text} for page in pages],
            },
            ensure_ascii=False,
        )
        try:
            result = await self.client.complete(ANALYSIS_SYSTEM_PROMPT, user_payload)
        except Exception as exc:
            raise DocumentAnalysisError(
                "Внешний анализ документа временно недоступен."
            ) from exc
        try:
            analysis = DocumentAnalysis.model_validate(
                _json_from_content(result.get("content", ""))
            )
        except ValidationError as exc:
            raise DocumentAnalysisError(
                "Ответ модели не прошёл проверку структуры."
            ) from exc
        return verify_analysis_citations(analysis, pages), "live"

    async def answer_question(
        self,
        pages: tuple[DocumentPage, ...],
        question: str,
        *,
        allow_external: bool,
    ) -> tuple[QuestionPayload, Literal["mock", "live", "disabled"]]:
        if _DIRECT_MEDICAL_ACTION.search(question):
            return (
                QuestionPayload(
                    answer=(
                        "Я могу найти в документе уже зафиксированные сведения, но не могу "
                        "назначать, отменять или изменять лечение."
                    ),
                    limitations=[
                        "Для медицинского решения обратитесь к лечащему врачу."
                    ],
                ),
                "disabled",
            )

        if not allow_external or not self.ready:
            tokens = {token.casefold() for token in _TOKEN.findall(question)}
            scored: list[tuple[int, int, str]] = []
            for page in pages:
                for line in page.text.splitlines():
                    clean = line.strip()
                    if not clean:
                        continue
                    score = sum(token in clean.casefold() for token in tokens)
                    if score:
                        scored.append((score, page.number, clean[:500]))
            scored.sort(key=lambda item: (-item[0], item[1]))
            evidence = [
                EvidenceItem(
                    category="other",
                    text=line,
                    page=page,
                    source_quote=line,
                )
                for _, page, line in scored[:5]
            ]
            mode: Literal["mock", "disabled"] = (
                "disabled" if not allow_external else "mock"
            )
            answer = (
                "Найдены релевантные фрагменты документа; проверьте цитаты ниже."
                if evidence
                else "Прямой ответ в тексте документа не найден."
            )
            return (
                QuestionPayload(
                    answer=answer,
                    evidence=evidence,
                    limitations=[
                        "Ответ сформирован локальным поиском без внешних медицинских знаний."
                    ],
                ),
                mode,
            )

        payload = json.dumps(
            {
                "question": question,
                "provenance": "uploaded_medical_pdf_untrusted",
                "pages": [{"page": page.number, "text": page.text} for page in pages],
            },
            ensure_ascii=False,
        )
        try:
            result = await self.client.complete(QUESTION_SYSTEM_PROMPT, payload)
        except Exception as exc:
            raise DocumentAnalysisError(
                "Ответ по документу временно недоступен."
            ) from exc
        try:
            parsed = QuestionPayload.model_validate(
                _json_from_content(result.get("content", ""))
            )
        except ValidationError as exc:
            raise DocumentAnalysisError(
                "Ответ модели не прошёл проверку структуры."
            ) from exc
        evidence, rejected = _verified_items(parsed.evidence, pages)
        limitations = list(parsed.limitations)
        if rejected:
            limitations.append(
                f"Исключено пунктов без проверяемой постраничной цитаты: {rejected}."
            )
        return parsed.model_copy(
            update={"evidence": evidence, "limitations": limitations}
        ), "live"
