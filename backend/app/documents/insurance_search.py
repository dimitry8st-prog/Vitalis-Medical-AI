from __future__ import annotations

import asyncio
import io
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings
from app.documents.models import (
    DocumentPage,
    InsuranceAnalysis,
    InsuranceWebLookup,
    OfficialSourceEvidence,
)
from app.documents.security import scan_untrusted_text
from app.rag.web_search import provider_search

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InsurerDefinition:
    name: str
    domain: str
    aliases: tuple[str, ...]


# Domains are deliberately allowlisted. A URL printed inside an uploaded PDF can never
# add a destination to this registry.
INSURERS: tuple[InsurerDefinition, ...] = (
    InsurerDefinition(
        "АльфаСтрахование-ОМС",
        "alfastrahoms.ru",
        ("альфастрахование-омс", "альфастрахование омс", "alfastrahoms.ru"),
    ),
    InsurerDefinition(
        "СОГАЗ-Мед",
        "sogaz-med.ru",
        ("согаз-мед", "согаз мед", "sogaz-med.ru"),
    ),
    InsurerDefinition(
        "Совкомбанк Страхование",
        "sovcomins.ru",
        ("совкомбанк страхование", "sovcomins.ru"),
    ),
    InsurerDefinition(
        "Ренессанс Страхование",
        "renins.ru",
        ("ренессанс страхование", "renins.ru"),
    ),
    InsurerDefinition(
        "Абсолют Страхование",
        "absolutins.ru",
        ("абсолют страхование", "absolutins.ru"),
    ),
    InsurerDefinition(
        "АльфаСтрахование",
        "alfastrah.ru",
        ("альфастрахование", "альфа страхование", "alfastrah.ru"),
    ),
    InsurerDefinition(
        "РЕСО-Гарантия",
        "reso.ru",
        ("ресо-гарантия", "ресо гарантия", "reso.ru"),
    ),
    InsurerDefinition(
        "Росгосстрах",
        "rgs.ru",
        ("росгосстрах", "rgs.ru"),
    ),
    InsurerDefinition(
        "Ингосстрах",
        "ingos.ru",
        ("ингосстрах", "ingos.ru"),
    ),
    InsurerDefinition(
        "Т-Страхование",
        "tinsurance.ru",
        ("т-страхование", "т страхование", "тинькофф страхование", "tinsurance.ru"),
    ),
    InsurerDefinition(
        "СберСтрахование",
        "sberbankins.ru",
        ("сберстрахование", "сбербанк страхование", "sberbankins.ru"),
    ),
    InsurerDefinition("СОГАЗ", "sogaz.ru", ("согаз", "sogaz.ru")),
    InsurerDefinition(
        "ВСК",
        "vsk.ru",
        ("страховой дом вск", "сао вск", "вск", "vsk.ru"),
    ),
)


_INSURANCE_TYPES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(?:дмс|DMS)\b|добровольн\w*\s+медицинск\w*"), "ДМС"),
    (re.compile(r"(?i)\b(?:омс|OMS)\b|обязательн\w*\s+медицинск\w*"), "ОМС"),
    (re.compile(r"(?i)\b(?:осаго|OSAGO)\b"), "ОСАГО"),
    (re.compile(r"(?i)\b(?:каско|KASKO)\b"), "КАСКО"),
    (
        re.compile(r"(?i)путешеств|выезжающ\w*\s+за\s+рубеж|туристическ\w*"),
        "Страхование путешественников",
    ),
    (re.compile(r"(?i)ипотечн\w*\s+страхован"), "Ипотечное страхование"),
    (re.compile(r"(?i)страхован\w*\s+(?:жизни|жизнь)"), "Страхование жизни"),
    (
        re.compile(r"(?i)несчастн\w*\s+случа|страхован\w*\s+здоров"),
        "Страхование от несчастных случаев и болезней",
    ),
    (
        re.compile(r"(?i)страхован\w*\s+(?:имуществ|квартир|дом)"),
        "Страхование имущества",
    ),
    (re.compile(r"(?i)титульн\w*\s+страхован"), "Титульное страхование"),
)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            value = re.sub(r"\s+", " ", data or "").strip()
            if value:
                self.parts.append(value)


def _all_text(pages: tuple[DocumentPage, ...]) -> str:
    return "\n".join(page.text for page in pages)


def detect_insurer(pages: tuple[DocumentPage, ...]) -> InsurerDefinition | None:
    text = _all_text(pages).casefold()
    for insurer in INSURERS:
        for alias in insurer.aliases:
            value = alias.casefold()
            if len(value) <= 3 and value.isalnum():
                if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text):
                    return insurer
            elif value in text:
                return insurer
    return None


def detect_insurance_type(pages: tuple[DocumentPage, ...]) -> str:
    text = _all_text(pages)
    for pattern, label in _INSURANCE_TYPES:
        if pattern.search(text):
            return label
    return "Не определён"


def is_official_insurer_url(url: str, domain: str) -> bool:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower().rstrip(".")
    allowed = domain.lower().rstrip(".")
    return (
        parsed.scheme == "https"
        and bool(host)
        and (host == allowed or host.endswith(f".{allowed}"))
        and parsed.username is None
        and parsed.password is None
    )


def _pdf_text(raw: bytes, max_pages: int = 60) -> str:
    reader = PdfReader(io.BytesIO(raw), strict=True)
    if reader.is_encrypted:
        return ""
    return "\n".join((page.extract_text() or "") for page in reader.pages[:max_pages])


def _html_text(raw: bytes, encoding: str | None = None) -> str:
    parser = _VisibleTextParser()
    parser.feed(raw.decode(encoding or "utf-8", errors="replace"))
    return "\n".join(parser.parts)


def select_insurance_excerpt(
    text: str,
    query: str,
    max_chars: int = 6_000,
) -> str:
    clean = re.sub(r"[ \t]+", " ", text or "")
    blocks = [
        block.strip()
        for block in re.split(r"\n{2,}|(?<=[.!?])\s+", clean)
        if block.strip()
    ]
    terms = list(
        dict.fromkeys(
            token.casefold()
            for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{2,}", query or "")
        )
    )[:12]
    insurance_terms = (
        "покры",
        "включ",
        "страхов",
        "риск",
        "случа",
        "исключ",
        "лимит",
        "франшиз",
        "выплат",
        "медицин",
        "амбулатор",
        "стационар",
    )
    ranked: list[tuple[int, int, str]] = []
    for index, block in enumerate(blocks):
        low = block.casefold()
        score = sum(3 for term in terms if term in low)
        score += sum(1 for term in insurance_terms if term in low)
        if score:
            ranked.append((score, -index, block))
    ranked.sort(reverse=True)

    chosen: list[str] = []
    size = 0
    for _, _, block in ranked:
        if block in chosen:
            continue
        remaining = max_chars - size
        if remaining <= 0:
            break
        chosen.append(block[:remaining])
        size += len(chosen[-1]) + 2
    return "\n\n".join(chosen)


async def fetch_official_excerpt(
    url: str,
    domain: str,
    query: str,
    settings: Settings,
) -> tuple[str, datetime | None]:
    """Fetch a bounded excerpt while keeping every redirect inside one allowlisted domain."""
    if not is_official_insurer_url(url, domain):
        return "", None

    current = url
    max_bytes = settings.insurance_web_max_bytes
    timeout = settings.insurance_web_timeout_seconds
    headers = {"User-Agent": "Vitalis-Insurance-Research/1.0"}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            for _ in range(4):
                async with client.stream("GET", current) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        target = urljoin(current, response.headers.get("location", ""))
                        if not is_official_insurer_url(target, domain):
                            return "", None
                        current = target
                        continue
                    response.raise_for_status()
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            return "", None
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    content_type = response.headers.get("content-type", "").lower()
                    if "pdf" in content_type or current.lower().endswith(".pdf"):
                        text = await asyncio.to_thread(_pdf_text, raw)
                    elif "html" in content_type or content_type.startswith("text/"):
                        text = _html_text(raw, response.encoding)
                    else:
                        return "", None
                    excerpt = select_insurance_excerpt(text, query, max_chars=6_000)
                    return excerpt, datetime.now(UTC)
    except (httpx.HTTPError, PdfReadError, ValueError, OSError, EOFError):
        return "", None
    return "", None


SearchProvider = Callable[[str, Settings, int], Awaitable[list[dict[str, Any]]]]
ExcerptFetcher = Callable[
    [str, str, str, Settings], Awaitable[tuple[str, datetime | None]]
]


class InsuranceOfficialSearch:
    def __init__(
        self,
        settings: Settings,
        *,
        search_provider: SearchProvider = provider_search,
        excerpt_fetcher: ExcerptFetcher = fetch_official_excerpt,
    ) -> None:
        self.settings = settings
        self.search_provider = search_provider
        self.excerpt_fetcher = excerpt_fetcher

    async def lookup_missing_coverage(
        self,
        pages: tuple[DocumentPage, ...],
        analysis: InsuranceAnalysis,
        *,
        requested: bool,
    ) -> InsuranceWebLookup:
        insurer = detect_insurer(pages)
        if not requested or not self.settings.insurance_official_search_enabled:
            return InsuranceWebLookup(status="not_requested")
        if analysis.coverage:
            return InsuranceWebLookup(
                status="not_needed",
                insurer_name=insurer.name if insurer else analysis.insurer_name,
                insurer_domain=insurer.domain if insurer else analysis.insurer_domain,
                warning="Сведения о покрытии найдены непосредственно в договоре.",
            )
        if insurer is None:
            return InsuranceWebLookup(
                status="unsupported_insurer",
                warning=(
                    "Страховщик не распознан или отсутствует в проверенном списке доменов; "
                    "автоматический веб-поиск не выполнялся."
                ),
            )

        insurance_type = detect_insurance_type(pages)
        query_type = (
            insurance_type if insurance_type != "Не определён" else "страховой полис"
        )
        query = (
            f"site:{insurer.domain} {query_type} что входит покрытие "
            "исключения правила страхования"
        )
        try:
            rows = await self.search_provider(
                query,
                self.settings,
                self.settings.insurance_web_max_results,
            )
        except Exception as exc:  # noqa: BLE001 - external providers must fail closed
            logger.warning(
                "insurance_search_provider_failed domain=%s error=%s",
                insurer.domain,
                type(exc).__name__,
            )
            rows = []
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            url = str(row.get("url") or "").strip()
            if not is_official_insurer_url(url, insurer.domain) or url in seen:
                continue
            seen.add(url)
            candidates.append(row)
            if len(candidates) >= self.settings.insurance_web_max_results:
                break
        if not candidates:
            candidates.append(
                {
                    "title": f"{insurer.name} — официальный сайт",
                    "url": f"https://{insurer.domain}/",
                    "snippet": "",
                }
            )

        sources: list[OfficialSourceEvidence] = []
        injection_omitted = False
        for row in candidates:
            url = str(row.get("url") or "")
            try:
                excerpt, retrieved_at = await self.excerpt_fetcher(
                    url,
                    insurer.domain,
                    query_type,
                    self.settings,
                )
            except Exception as exc:  # noqa: BLE001 - one failed page is non-fatal
                logger.warning(
                    "insurance_official_fetch_failed domain=%s error=%s",
                    insurer.domain,
                    type(exc).__name__,
                )
                excerpt, retrieved_at = "", None
            if excerpt:
                report = scan_untrusted_text((DocumentPage(number=1, text=excerpt),))
                if report.prompt_injection_detected:
                    excerpt = ""
                    injection_omitted = True
            title = re.sub(r"\s+", " ", str(row.get("title") or insurer.name)).strip()
            snippet = re.sub(r"\s+", " ", str(row.get("snippet") or "")).strip()
            if not snippet and not excerpt:
                continue
            sources.append(
                OfficialSourceEvidence(
                    title=(title or insurer.name)[:500],
                    url=url,
                    domain=insurer.domain,
                    snippet=snippet[:2_000],
                    evidence_excerpt=excerpt[:6_000],
                    retrieved_at=retrieved_at,
                )
            )

        if not sources:
            return InsuranceWebLookup(
                status="not_found",
                insurer_name=insurer.name,
                insurer_domain=insurer.domain,
                query=query,
                warning=(
                    "В договоре покрытие не найдено; на официальном сайте страховщика "
                    "подходящий материал автоматически не найден."
                ),
            )
        warning = (
            "Данные официального сайта являются общей справкой: применяются условия, "
            "редакция правил и программа, указанные именно в вашем договоре."
        )
        if injection_omitted:
            warning += " Небезопасный фрагмент одной из страниц не использован."
        return InsuranceWebLookup(
            status="found",
            insurer_name=insurer.name,
            insurer_domain=insurer.domain,
            query=query,
            sources=sources,
            warning=warning,
        )
