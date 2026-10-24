from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from app.documents.insurance_search import detect_insurance_type, detect_insurer
from app.documents.models import (
    AnalysisPayload,
    DocumentAnalysis,
    DocumentPage,
    EvidenceItem,
    InsuranceAnalysis,
    OfficialSourceEvidence,
)
from app.documents.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    INSURANCE_ANALYSIS_SYSTEM_PROMPT,
    INSURANCE_QUESTION_SYSTEM_PROMPT,
    QUESTION_SYSTEM_PROMPT,
)
from app.documents.redactor import redact_pages
from app.documents.security import scan_untrusted_text
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
    official_sources: list[OfficialSourceEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


_DIRECT_MEDICAL_ACTION = re.compile(
    r"(?i)\b(?:назначь|назначить|отменить|начать\s+принимать|что\s+принимать|как\s+лечить|"
    r"подбери\s+лечени|измени\s+доз|дозировк[ау]\s+для\s+меня)"
)
_LEGAL_OR_CLAIM_COMMITMENT = re.compile(
    r"(?i)\b(?:гарантируй\s+выплат|обязана\s+выплатить|точно\s+выплатят|"
    r"дай\s+юридическ\w*\s+заключ|подай\s+(?:за\s+меня\s+)?заявлен|"
    r"признай\s+страхов\w*\s+случа)"
)
_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{4,}")
_LABELLED_LINE = re.compile(
    r"(?i)^(?:диагноз|заключение|результат|рекомендации?|назначения?)\s*[:—-].+"
)
_MEASUREMENT = re.compile(
    r"(?i)\b\d+(?:[.,]\d+)?\s*(?:ммоль/л|мкмоль/л|г/л|мг/л|ед/л|мм\s*рт\.?\s*ст\.?)\b"
)
_INSURANCE_SECTIONS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "coverage",
        "coverage",
        re.compile(r"(?i)страхов\w*\s+(?:риск|случа)|покрыва|включен|входит"),
    ),
    (
        "exclusions",
        "exclusion",
        re.compile(r"(?i)исключен|не\s+явля\w*\s+страхов|не\s+покрыва"),
    ),
    (
        "limits",
        "limit",
        re.compile(r"(?i)страхов\w*\s+сумм|лимит|не\s+более|максимальн\w*\s+выплат"),
    ),
    ("deductibles", "deductible", re.compile(r"(?i)франшиз")),
    (
        "validity",
        "validity",
        re.compile(r"(?i)срок\s+действ|период\s+страхован|действует\s+с"),
    ),
    (
        "claim_procedure",
        "claim_procedure",
        re.compile(r"(?i)при\s+наступлен|обратит|уведомить|заявлен\w*\s+о\s+страхов"),
    ),
    ("contacts", "contact", re.compile(r"(?i)контакт|телефон|горяч\w*\s+лини|сайт")),
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
            # A real quote cannot validate a different, model-invented clinical fact.
            accepted.append(item.model_copy(update={"text": item.source_quote}))
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


def verify_insurance_citations(
    analysis: InsuranceAnalysis,
    pages: tuple[DocumentPage, ...],
) -> InsuranceAnalysis:
    updates: dict[str, Any] = {}
    rejected = 0
    evidence_fields = (
        "coverage",
        "exclusions",
        "limits",
        "deductibles",
        "validity",
        "claim_procedure",
        "contacts",
    )
    for field_name in evidence_fields:
        verified, count = _verified_items(getattr(analysis, field_name), pages)
        updates[field_name] = verified
        rejected += count

    insurer = detect_insurer(pages)
    insurance_type = detect_insurance_type(pages)
    full_text = _normalise_quote(" ".join(page.text for page in pages))
    policy_name = (analysis.policy_name or "").strip()
    verified_policy_name = (
        policy_name
        if policy_name and _normalise_quote(policy_name) in full_text
        else None
    )
    if policy_name and verified_policy_name is None:
        rejected += 1
    updates["document_type"] = "Страховой договор или полис"
    updates["insurer_name"] = insurer.name if insurer else None
    updates["insurer_domain"] = insurer.domain if insurer else None
    updates["insurance_type"] = insurance_type
    updates["policy_name"] = verified_policy_name

    summary_evidence = [
        item
        for field_name in ("coverage", "limits", "validity", "exclusions")
        for item in updates[field_name]
    ][:4]
    if summary_evidence:
        quotes = " ".join(f"«{item.source_quote}»" for item in summary_evidence)
        updates["summary"] = f"Ключевые проверяемые условия договора: {quotes}"
    else:
        updates["summary"] = (
            "Условия страхового покрытия с проверяемыми цитатами не найдены. "
            "Проверьте договор и правила страхования вручную."
        )
    updates["limitations"] = [
        *analysis.limitations,
        "Выжимка сформирована только из цитат, найденных на указанных страницах.",
        "Итоговая применимость условий определяется договором и его редакцией правил.",
    ]
    if rejected:
        updates["limitations"].append(
            f"Исключено пунктов без проверяемой постраничной цитаты: {rejected}."
        )
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


def _offline_insurance_analysis(
    pages: tuple[DocumentPage, ...],
    mode: Literal["mock", "disabled"],
) -> InsuranceAnalysis:
    extracted: dict[str, list[EvidenceItem]] = {
        field_name: [] for field_name, _, _ in _INSURANCE_SECTIONS
    }
    for page in pages:
        for line in page.text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            for field_name, category, pattern in _INSURANCE_SECTIONS:
                if field_name == "coverage" and _INSURANCE_SECTIONS[1][2].search(clean):
                    continue
                if pattern.search(clean) and len(extracted[field_name]) < 12:
                    extracted[field_name].append(
                        EvidenceItem(
                            category=category,
                            text=clean[:500],
                            page=page.number,
                            source_quote=clean[:500],
                        )
                    )
                    break

    insurer = detect_insurer(pages)
    insurance_type = detect_insurance_type(pages)
    reason = (
        "Внешний AI-анализ заблокирован проверкой безопасности; показано только "
        "детерминированное извлечение."
        if mode == "disabled"
        else "Внешний AI-анализ выключен; показано только детерминированное извлечение."
    )
    missing: list[str] = []
    if not extracted["coverage"]:
        missing.append("Состав страхового покрытия не найден в тексте PDF.")
    if insurance_type == "Не определён":
        missing.append("Тип страхования не удалось определить автоматически.")
    if insurer is None:
        missing.append(
            "Страховую компанию не удалось сопоставить с проверенным доменом."
        )
    return InsuranceAnalysis(
        document_type="Страховой договор или полис",
        summary=(
            "PDF проверен, обезличен и разобран по условиям страхования. "
            "Показаны только строки, которые можно сверить с исходным договором."
        ),
        insurer_name=insurer.name if insurer else None,
        insurer_domain=insurer.domain if insurer else None,
        insurance_type=insurance_type,
        coverage=extracted["coverage"],
        exclusions=extracted["exclusions"],
        limits=extracted["limits"],
        deductibles=extracted["deductibles"],
        validity=extracted["validity"],
        claim_procedure=extracted["claim_procedure"],
        contacts=extracted["contacts"],
        missing_information=missing,
        questions_for_insurer=[
            "Какая редакция правил страхования применяется к этому полису?",
            "Какие исключения и лимиты действуют для выбранной программы?",
        ],
        limitations=[
            reason,
            "Разбор не является юридическим заключением и не гарантирует выплату.",
            "Сканированные страницы без текстового слоя не распознаются.",
        ],
    )


def _question_tokens(question: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(question)}


def _local_question_evidence(
    pages: tuple[DocumentPage, ...],
    question: str,
    document_kind: Literal["medical", "insurance"] = "medical",
) -> list[EvidenceItem]:
    tokens = _question_tokens(question)
    intent_pattern: re.Pattern[str] | None = None
    if document_kind == "insurance":
        low = question.casefold()
        if "тип" in low or "вид" in low:
            tokens.update({"дмс", "омс", "осаго", "каско", "страхован"})
        if "исключ" in low or "не покры" in low:
            tokens.update({"исключ", "покры", "страхов"})
            intent_pattern = _INSURANCE_SECTIONS[1][2]
        elif "входит" in low or "покры" in low or "включ" in low:
            tokens.update({"покры", "включ", "страхов", "риск"})
            intent_pattern = _INSURANCE_SECTIONS[0][2]
        elif "лимит" in low or "страхов" in low and "сумм" in low:
            tokens.update({"лимит", "сумм", "выплат"})
            intent_pattern = _INSURANCE_SECTIONS[2][2]
        elif "франшиз" in low:
            tokens.add("франшиз")
            intent_pattern = _INSURANCE_SECTIONS[3][2]
        elif "срок" in low or "период" in low or "действует" in low:
            tokens.update({"срок", "период", "действует"})
            intent_pattern = _INSURANCE_SECTIONS[4][2]
    scored: list[tuple[int, int, str]] = []
    for page in pages:
        for line in page.text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if intent_pattern is not None and not intent_pattern.search(clean):
                continue
            if intent_pattern is _INSURANCE_SECTIONS[0][2] and _INSURANCE_SECTIONS[1][
                2
            ].search(clean):
                continue
            score = sum(token in clean.casefold() for token in tokens)
            if score:
                scored.append((score, page.number, clean[:500]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        EvidenceItem(
            category="other",
            text=line,
            page=page,
            source_quote=line,
        )
        for _, page, line in scored[:5]
    ]


def _relevant_official_sources(
    question: str,
    sources: list[OfficialSourceEvidence],
) -> list[OfficialSourceEvidence]:
    tokens = _question_tokens(question)
    ranked: list[tuple[int, OfficialSourceEvidence]] = []
    for source in sources:
        text = (f"{source.title} {source.snippet} {source.evidence_excerpt}").casefold()
        score = sum(token in text for token in tokens)
        ranked.append((score, source))
    ranked.sort(key=lambda item: item[0], reverse=True)
    relevant = [source for score, source in ranked if score > 0]
    return (relevant or [source for _, source in ranked])[:3]


class DocumentAnalyzer:
    def __init__(self, settings: Settings, client: CompletionClient | None = None):
        self.settings = settings
        self.client = client or DeepSeekClient(settings)

    @property
    def ready(self) -> bool:
        return self.settings.document_external_llm_ready and self.client.ready

    async def analyze(
        self,
        pages: tuple[DocumentPage, ...],
        *,
        allow_external: bool = True,
        document_kind: Literal["medical", "insurance"] = "medical",
    ) -> tuple[AnalysisPayload, Literal["mock", "live", "disabled"]]:
        pages, pii = redact_pages(pages)
        allow_external = (
            allow_external
            and not pii.review_required
            and not scan_untrusted_text(pages).prompt_injection_detected
        )
        offline = (
            _offline_insurance_analysis
            if document_kind == "insurance"
            else _offline_analysis
        )
        if not allow_external:
            return offline(pages, "disabled"), "disabled"
        if not self.ready:
            return offline(pages, "mock"), "mock"

        provenance = f"uploaded_{document_kind}_pdf_untrusted"
        user_payload = json.dumps(
            {
                "provenance": provenance,
                "pages": [{"page": page.number, "text": page.text} for page in pages],
            },
            ensure_ascii=False,
        )
        prompt = (
            INSURANCE_ANALYSIS_SYSTEM_PROMPT
            if document_kind == "insurance"
            else ANALYSIS_SYSTEM_PROMPT
        )
        try:
            result = await self.client.complete(prompt, user_payload)
        except Exception as exc:
            raise DocumentAnalysisError(
                "Внешний анализ документа временно недоступен."
            ) from exc
        try:
            payload = _json_from_content(result.get("content", ""))
            if document_kind == "insurance":
                insurance_analysis = InsuranceAnalysis.model_validate(payload)
                return verify_insurance_citations(insurance_analysis, pages), "live"
            analysis = DocumentAnalysis.model_validate(payload)
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
        document_kind: Literal["medical", "insurance"] = "medical",
        official_sources: list[OfficialSourceEvidence] | None = None,
    ) -> tuple[QuestionPayload, Literal["mock", "live", "disabled"]]:
        # Follow-up text is a fresh untrusted input, even for an approved PDF session.
        pages, page_pii = redact_pages(pages)
        question_pages, question_pii = redact_pages(
            (DocumentPage(number=1, text=question),)
        )
        question = question_pages[0].text
        allow_external = (
            allow_external
            and not page_pii.review_required
            and not question_pii.review_required
            and not scan_untrusted_text(
                (*pages, *question_pages)
            ).prompt_injection_detected
        )
        official_sources = official_sources or []
        if document_kind == "medical" and _DIRECT_MEDICAL_ACTION.search(question):
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
        if document_kind == "insurance" and _LEGAL_OR_CLAIM_COMMITMENT.search(question):
            return (
                QuestionPayload(
                    answer=(
                        "Я могу сопоставить условия договора и материалы официального "
                        "сайта страховщика, но не могу гарантировать выплату, признать "
                        "случай страховым или подать заявление от вашего имени."
                    ),
                    limitations=[
                        "Юридически значимое решение принимает страховщик; при споре нужна проверка специалистом."
                    ],
                ),
                "disabled",
            )

        if not allow_external or not self.ready:
            evidence = _local_question_evidence(pages, question, document_kind)
            web_evidence = (
                _relevant_official_sources(question, official_sources)
                if document_kind == "insurance" and not evidence
                else []
            )
            mode: Literal["mock", "disabled"] = (
                "disabled" if not allow_external else "mock"
            )
            if evidence:
                answer = (
                    "Найдены релевантные фрагменты документа; проверьте цитаты ниже."
                )
            elif web_evidence:
                answer = (
                    "Прямой ответ в договоре не найден. Ниже показаны релевантные "
                    "материалы только с официального сайта распознанного страховщика."
                )
            else:
                answer = "Прямой ответ в тексте документа не найден."
            limitation = (
                "Ответ сформирован локальным поиском; материалы сайта не заменяют условия конкретного договора."
                if document_kind == "insurance"
                else "Ответ сформирован локальным поиском без внешних медицинских знаний."
            )
            return (
                QuestionPayload(
                    answer=answer,
                    evidence=evidence,
                    official_sources=web_evidence,
                    limitations=[limitation],
                ),
                mode,
            )

        provenance = f"uploaded_{document_kind}_pdf_untrusted"
        payload = json.dumps(
            {
                "question": question,
                "provenance": provenance,
                "pages": [{"page": page.number, "text": page.text} for page in pages],
            },
            ensure_ascii=False,
        )
        prompt = (
            INSURANCE_QUESTION_SYSTEM_PROMPT
            if document_kind == "insurance"
            else QUESTION_SYSTEM_PROMPT
        )
        try:
            result = await self.client.complete(prompt, payload)
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
        web_evidence = (
            _relevant_official_sources(question, official_sources)
            if document_kind == "insurance" and not evidence
            else []
        )
        if evidence:
            quotes = " ".join(f"«{item.source_quote}»" for item in evidence[:4])
            answer = f"Проверяемые фрагменты документа: {quotes}"
        else:
            answer = "Прямой ответ в тексте документа не найден."
        if web_evidence:
            answer = (
                f"{answer} Дополнительно показаны материалы официального сайта; "
                "они не заменяют условия конкретного договора."
            )
        return parsed.model_copy(
            update={
                "answer": answer,
                "evidence": evidence,
                "official_sources": web_evidence,
                "limitations": limitations,
            }
        ), "live"
