from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx

from app.config import Settings


async def web_search(
    query: str,
    settings: Settings,
    max_results: int = 4,
    bilingual: bool = True,
    freshness_extra: bool = False,
) -> list[dict[str, Any]]:
    """Сначала PubMed (стабильно), DDG — только если мало результатов."""
    provider = settings.web_search_provider
    all_hits: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Международный контур: PubMed по английскому варианту запроса.
    for batch, lang in await _fallback_bundle(query, max_results):
        _merge(all_hits, seen, batch, lang, "pubmed")

    # RU и EN выполняются независимо: наличие одного языка не отменяет второй.
    queries = _build_queries(query, bilingual=bilingual, freshness_extra=freshness_extra)

    async def run_one(item: tuple[str, str, str]) -> tuple[list[dict[str, Any]], str, str]:
        q, lang, kind = item
        try:
            batch = await asyncio.wait_for(
                _run_provider(provider, q, settings, max_results),
                timeout=6.0,
            )
        except Exception:
            batch = []
        return batch, lang, kind

    batches = await asyncio.gather(*(run_one(item) for item in queries))
    for batch, lang, kind in batches:
        _merge(all_hits, seen, batch, lang, kind)

    all_hits.sort(key=lambda x: (x.get("year") or 0), reverse=True)
    return all_hits[: max(max_results * 3, 6)]


def _merge(
    all_hits: list[dict[str, Any]],
    seen: set[str],
    batch: list[dict[str, Any]],
    lang: str,
    kind: str,
) -> None:
    for item in batch:
        if item.get("provider") == "error":
            continue
        item = dict(item)
        item["lang"] = lang
        item["query_kind"] = kind
        url = (item.get("url") or "").strip()
        key = url or f"{item.get('title')}|{item.get('snippet')}"
        if not key or key in seen:
            continue
        seen.add(key)
        item["year"] = _extract_year(f"{item.get('title','')} {item.get('snippet','')}")
        all_hits.append(item)


def _build_queries(
    query: str,
    bilingual: bool,
    freshness_extra: bool,
) -> list[tuple[str, str, str]]:
    year = __import__("datetime").datetime.now().year
    prev = year - 1
    ru = f"{query} клинические рекомендации Минздрав"
    en_query = _english_variant(query)
    en = f"{en_query} clinical guidelines"
    out: list[tuple[str, str, str]] = [(ru, "ru", "base")]
    if bilingual:
        out.append((en, "en", "base"))
    if freshness_extra:
        out.append((f"{query} guidelines {prev} OR {year}", "en", "freshness"))
        out.append((f"{query} рекомендации {prev} OR {year}", "ru", "freshness"))
    return out


async def _run_provider(
    provider: str,
    query: str,
    settings: Settings,
    max_results: int,
) -> list[dict[str, Any]]:
    try:
        if provider == "serper" and settings.serper_api_key:
            return await _serper(query, settings.serper_api_key, max_results)
        if provider == "tavily" and settings.tavily_api_key:
            return await _tavily(query, settings.tavily_api_key, max_results)
        return await _duckduckgo(query, max_results)
    except Exception:
        return []


_RU_EN_TERMS = {
    "туберкулез": "tuberculosis",
    "туберкулёз": "tuberculosis",
    "геморрой": "hemorrhoids",
    "пневмония": "pneumonia",
    "гипертензия": "hypertension",
    "гипертония": "hypertension",
    "инфаркт": "myocardial infarction",
    "окс": "acute coronary syndrome",
    "диабет": "diabetes mellitus",
    " астма": "asthma",
    "бронхит": "bronchitis",
    "перелом": "fracture",
    "позвоночник": "spine fracture",
    "онкология": "oncology cancer",
    "инсульт": "stroke",
    "чмт": "traumatic brain injury",
    "черепно-мозгов": "traumatic brain injury",
    "черепно мозгов": "traumatic brain injury",
}


def _english_variant(query: str) -> str:
    low = (query or "").lower()
    for ru, en in _RU_EN_TERMS.items():
        if ru.strip() in low:
            return en
    return query


def _search_variants(query: str) -> list[str]:
    q = (query or "").strip()
    low = q.lower()
    variants = [q]
    # убираем вопросительные оболочки
    cleaned = re.sub(
        r"(?i)^(что\s+такое|как\s+лечить|как\s+диагностир\w*|протоколы?\s+|лечение\s+|диагностика\s+)\s*",
        "",
        q,
    ).strip(" ?!.")
    if cleaned and cleaned.lower() != low:
        variants.append(cleaned)
    for ru, en in _RU_EN_TERMS.items():
        if ru.strip() in low:
            variants.append(en)
            variants.append(f"{en} clinical guidelines")
            variants.append(f"{cleaned or q} clinical recommendations")
            break
    # уникальные, непустые
    out, seen = [], set()
    for v in variants:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out[:5]


async def _fallback_bundle(
    query: str, max_results: int
) -> list[tuple[list[dict[str, Any]], str]]:
    variants = _search_variants(query)
    # PubMed лучше с EN: сначала английские варианты
    en_first = sorted(
        variants,
        key=lambda v: (0 if re.search(r"[A-Za-z]", v) else 1, len(v)),
    )
    out: list[tuple[list[dict[str, Any]], str]] = []
    for v in en_first:
        batch = await _pubmed(v, max_results)
        if batch:
            out.append((batch, "en"))
            break  # одного удачного EN-запроса достаточно
    # если EN не сработал — попробовать очищенный оригинал
    if not out:
        for v in variants:
            batch = await _pubmed(v, max_results)
            if batch:
                out.append((batch, "en"))
                break
    return out


async def _wikipedia(query: str, lang: str, max_results: int = 2) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=12, headers=headers) as client:
            res = await client.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": max_results,
                    "namespace": 0,
                    "format": "json",
                },
            )
            if res.status_code != 200:
                return []
            data = res.json()
    except Exception:
        return []

    if not isinstance(data, list) or len(data) < 4:
        return []
    titles, descs, urls = data[1], data[2], data[3]
    return [
        {
            "title": f"Wikipedia: {title}",
            "url": url,
            "snippet": desc or title,
            "provider": "wikipedia",
        }
        for title, desc, url in zip(titles, descs, urls)
    ]


def _extract_year(text: str) -> int | None:
    years = [int(y) for y in re.findall(r"\b(20[1-2]\d)\b", text or "")]
    return max(years) if years else None


async def _duckduckgo(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []

    def _sync() -> list[dict[str, Any]]:
        try:
            ddgs = DDGS()
            rows = list(ddgs.text(query, max_results=max_results) or [])
            return [
                {
                    "title": r.get("title") or "",
                    "url": r.get("href") or r.get("link") or "",
                    "snippet": r.get("body") or r.get("snippet") or "",
                    "provider": "duckduckgo",
                }
                for r in rows
            ]
        except Exception:
            return []

    try:
        return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=5.0)
    except Exception:
        return []


async def _pubmed(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            es = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                    "tool": "vitalis",
                    "email": "vitalis@local.dev",
                },
            )
            es.raise_for_status()
            ids = es.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            sm = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            )
            sm.raise_for_status()
            result = sm.json().get("result", {})
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for pmid in ids:
        item = result.get(pmid) or {}
        title = item.get("title") or f"PubMed {pmid}"
        pubdate = item.get("pubdate") or item.get("epubdate") or ""
        source = item.get("fulljournalname") or item.get("source") or "PubMed"
        out.append(
            {
                "title": f"{title} ({source}, {pubdate})".strip(),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "snippet": f"PubMed PMID {pmid}. {source}. {pubdate}",
                "provider": "pubmed",
            }
        )
    return out


async def _serper(query: str, api_key: str, max_results: int) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20) as client:
        res = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
        )
        res.raise_for_status()
        data = res.json()
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "provider": "serper",
        }
        for item in data.get("organic", [])[:max_results]
    ]


async def _tavily(query: str, api_key: str, max_results: int) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20) as client:
        res = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
        )
        res.raise_for_status()
        data = res.json()
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
            "provider": "tavily",
        }
        for item in data.get("results", [])[:max_results]
    ]
