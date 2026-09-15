from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import httpx

from app.config import Settings


async def web_search(
    query: str,
    settings: Settings,
    max_results: int = 4,
    bilingual: bool = True,
    freshness_extra: bool = False,
    scope: str = "both",
) -> list[dict[str, Any]]:
    """Search by clinical concepts, keeping RF and international evidence separate."""
    provider = settings.web_search_provider
    all_hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    focus = _extract_focus(query)

    if scope in ("intl", "both"):
        for batch, lang, kind in await _literature_bundle(query, max_results):
            _merge(all_hits, seen, batch, lang, kind)

    queries = _build_queries(
        query,
        bilingual=bilingual,
        freshness_extra=freshness_extra,
        scope=scope,
    )
    semaphore = asyncio.Semaphore(3)

    async def run_one(item: tuple[str, str, str]) -> tuple[list[dict[str, Any]], str, str]:
        q, lang, kind = item
        async with semaphore:
            try:
                batch = await asyncio.wait_for(
                    _run_provider(provider, q, settings, max_results),
                    timeout=12.0,
                )
            except Exception:
                batch = []
        return batch, lang, kind

    batches = await asyncio.gather(*(run_one(item) for item in queries))
    for batch, lang, kind in batches:
        _merge(all_hits, seen, batch, lang, kind)

    for item in all_hits:
        item["relevance_score"] = _score_hit(item, focus)
    return _balanced_results(all_hits, scope, max_results)


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
    scope: str = "both",
) -> list[tuple[str, str, str]]:
    year = datetime.now().year
    prev = year - 1
    focus = _extract_focus(query)
    en_focus = _english_variant(focus)
    intent_ru, intent_en = _detect_intent(query)
    out: list[tuple[str, str, str]] = []

    if scope in ("rf", "both"):
        out.extend(
            [
                (f'"{focus}" клинические рекомендации Минздрав', "ru", "rf_guideline"),
                (f'site:cr.minzdrav.gov.ru "{focus}"', "ru", "rf_registry"),
                (f'"{focus}" НМИЦ {intent_ru} клинический обзор', "ru", "rf_institute"),
                (f'"{focus}" {intent_ru} медицинский журнал обзор рекомендации', "ru", "rf_journal"),
            ]
        )
        if freshness_extra:
            out.append(
                (f'"{focus}" клинические рекомендации {prev} OR {year}', "ru", "rf_freshness")
            )

    if bilingual and scope in ("intl", "both"):
        out.extend(
            [
                (f'"{en_focus}" clinical practice guideline {intent_en}', "en", "intl_guideline"),
                (f'"{en_focus}" guideline WHO NICE European society {intent_en}', "en", "intl_europe"),
                (f'"{en_focus}" guideline NIH CDC American society {intent_en}', "en", "intl_us"),
                (f'"{en_focus}" systematic review meta-analysis consensus {intent_en}', "en", "intl_literature"),
            ]
        )
        if freshness_extra:
            out.append(
                (f'"{en_focus}" guideline {prev} OR {year}', "en", "intl_freshness")
            )

    deduped: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for item in out:
        key = item[0].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


async def _run_provider(
    provider: str,
    query: str,
    settings: Settings,
    max_results: int,
) -> list[dict[str, Any]]:
    primary: list[dict[str, Any]] = []
    try:
        if provider == "serper" and settings.serper_api_key:
            primary = await _serper(query, settings.serper_api_key, max_results)
        elif provider == "tavily" and settings.tavily_api_key:
            primary = await _tavily(query, settings.tavily_api_key, max_results)
        else:
            primary = await _duckduckgo(query, max_results)
    except Exception:
        primary = []

    if primary:
        return primary
    return await _bing_rss(query, max_results)


_RU_EN_TERMS = {
    "ишемический инсульт": "ischemic stroke",
    "геморрагический инсульт": "hemorrhagic stroke",
    "черепно-мозговая травма": "traumatic brain injury",
    "черепно мозговая травма": "traumatic brain injury",
    "рассеянный склероз": "multiple sclerosis",
    "болезнь паркинсона": "Parkinson disease",
    "болезнь альцгеймера": "Alzheimer disease",
    "сердечная недостаточность": "heart failure",
    "фибрилляция предсердий": "atrial fibrillation",
    "хроническая болезнь почек": "chronic kidney disease",
    "хроническая обструктивная болезнь легких": "chronic obstructive pulmonary disease",
    "рак молочной железы": "breast cancer",
    "рак предстательной железы": "prostate cancer",
    "колоректальный рак": "colorectal cancer",
    "рак легкого": "lung cancer",
    "туберкулез": "tuberculosis",
    "туберкулёз": "tuberculosis",
    "геморрой": "hemorrhoids",
    "пневмония": "pneumonia",
    "гипертензия": "hypertension",
    "гипертония": "hypertension",
    "инфаркт": "myocardial infarction",
    "окс": "acute coronary syndrome",
    "диабет": "diabetes mellitus",
    "астма": "asthma",
    "бронхит": "bronchitis",
    "перелом": "fracture",
    "позвоночник": "spine",
    "онкология": "oncology",
    "инсульт": "stroke",
    "эпилепсия": "epilepsy",
    "мигрень": "migraine",
    "цирроз": "cirrhosis",
    "гепатит": "hepatitis",
    "сепсис": "sepsis",
    "вич": "HIV",
    "чмт": "traumatic brain injury",
}

_INTENT_TERMS = (
    (r"лечен|терап|препарат|дозиров", "лечение", "treatment"),
    (r"диагност|обследован|скрининг", "диагностика", "diagnosis"),
    (r"реабилитац|восстанов", "реабилитация", "rehabilitation"),
    (r"профилактик|предотвращ", "профилактика", "prevention"),
    (r"классификац|стади|степен", "классификация", "classification"),
    (r"прогноз|исход", "прогноз", "prognosis"),
)


def _detect_intent(query: str) -> tuple[str, str]:
    low = (query or "").lower()
    for pattern, ru, en in _INTENT_TERMS:
        if re.search(pattern, low):
            return ru, en
    return "ведение", "management"


def _extract_focus(query: str) -> str:
    text = re.sub(r"\s+", " ", (query or "").strip())
    text = re.sub(
        r"(?i)^(что\s+такое|как\s+лечить|как\s+диагностир\w*|"
        r"расскажи(?:те)?\s+про|дай(?:те)?\s+рекомендации\s+по|"
        r"классификация|лечение|диагностика|реабилитация)\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)\b(по\s+актуальным\s+рекомендациям|по\s+клиническим\s+рекомендациям|"
        r"рф|международные|российские|клинические\s+рекомендации)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip(" ?!.,:;")
    return text or (query or "").strip()


def _english_variant(query: str) -> str:
    low = (query or "").lower()
    matched: list[str] = []
    for ru, en in sorted(_RU_EN_TERMS.items(), key=lambda pair: len(pair[0]), reverse=True):
        if ru in low and en not in matched:
            matched.append(en)
    codes = re.findall(r"\b[A-Z]\d{2}(?:\.\d+)?\b", query or "", flags=re.I)
    if matched:
        return " ".join(matched + codes)
    latin = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query or "")
    return " ".join(latin + codes) or _extract_focus(query)

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


async def _literature_bundle(
    query: str, max_results: int
) -> list[tuple[list[dict[str, Any]], str, str]]:
    focus = _english_variant(_extract_focus(query))
    _, intent_en = _detect_intent(query)
    current_year = datetime.now().year
    pubmed_query = (
        f'("{focus}"[Title/Abstract]) AND '
        f'(guideline[Publication Type] OR practice guideline[Publication Type] '
        f'OR systematic review[Publication Type] OR consensus[Title/Abstract]) AND '
        f'("2021/01/01"[Date - Publication] : "{current_year}/12/31"[Date - Publication])'
    )
    europe_query = (
        f'"{focus}" AND ({intent_en} OR guideline OR consensus OR systematic review) '
        f'AND FIRST_PDATE:[2021 TO {current_year}]'
    )
    pubmed, europe = await asyncio.gather(
        _pubmed(pubmed_query, max_results),
        _europe_pmc(europe_query, max_results),
    )
    out: list[tuple[list[dict[str, Any]], str, str]] = []
    if pubmed:
        out.append((pubmed, "en", "pubmed"))
    if europe:
        out.append((europe, "en", "europe_pmc"))
    return out


async def _fallback_bundle(
    query: str, max_results: int
) -> list[tuple[list[dict[str, Any]], str]]:
    """Backward-compatible wrapper used by older callers."""
    return [(batch, lang) for batch, lang, _ in await _literature_bundle(query, max_results)]


def _score_hit(item: dict[str, Any], focus: str) -> float:
    url = str(item.get("url") or "").lower()
    haystack = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    words = {
        word for word in re.findall(r"[a-zа-яё0-9]{4,}", focus.lower())
        if word not in {"клинические", "рекомендации", "лечение", "диагностика"}
    }
    overlap = sum(1 for word in words if word in haystack)
    authority = 0
    tiers = (
        (5, ("cr.minzdrav.gov.ru", "minzdrav.gov.ru", "who.int", "nice.org.uk")),
        (4, ("nih.gov", "cdc.gov", "pubmed.ncbi.nlm.nih.gov", "europepmc.org", "cochranelibrary.com")),
        (3, ("escardio.org", "heart.org", "acc.org", "idsociety.org", "nccn.org", "eular.org", "ersnet.org")),
        (2, ("sechenov.ru", "rsmu.ru", "nmicr.ru", "oncology.ru", "mediasphera.ru", "rnmot.ru")),
    )
    for points, domains in tiers:
        if any(domain in url for domain in domains):
            authority = points
            break
    year = item.get("year") or 0
    freshness = 2 if year >= datetime.now().year - 2 else 1 if year >= datetime.now().year - 5 else 0
    return float(overlap * 2 + authority + freshness)


def _balanced_results(
    hits: list[dict[str, Any]], scope: str, max_results: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        hits,
        key=lambda item: (item.get("relevance_score", 0), item.get("year") or 0),
        reverse=True,
    )
    if scope == "rf":
        return [item for item in ranked if item.get("lang") == "ru"][: max_results * 2]
    if scope == "intl":
        return [item for item in ranked if item.get("lang") == "en"][: max_results * 2]
    ru = [item for item in ranked if item.get("lang") == "ru"][:max_results]
    en = [item for item in ranked if item.get("lang") == "en"][:max_results]
    return sorted(
        ru + en,
        key=lambda item: (item.get("relevance_score", 0), item.get("year") or 0),
        reverse=True,
    )


async def _europe_pmc(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": max_results,
                    "resultType": "core",
                },
            )
            response.raise_for_status()
            rows = response.json().get("resultList", {}).get("result", [])
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        pmid = row.get("pmid")
        doi = row.get("doi")
        url = (
            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            if pmid
            else f"https://europepmc.org/article/MED/{row.get('id', '')}"
        )
        out.append(
            {
                "title": row.get("title") or "Europe PMC",
                "url": url,
                "snippet": " ".join(
                    str(value)
                    for value in (
                        row.get("authorString"),
                        row.get("journalTitle"),
                        row.get("pubYear"),
                        f"DOI {doi}" if doi else "",
                    )
                    if value
                ),
                "provider": "europe_pmc",
            }
        )
    return out


async def _bing_rss(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 Vitalis/1.0"},
            follow_redirects=True,
        ) as client:
            response = await client.get(
                "https://www.bing.com/search",
                params={"q": query, "format": "rss", "setlang": "ru"},
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for node in root.findall(".//item")[:max_results]:
        out.append(
            {
                "title": node.findtext("title") or "",
                "url": node.findtext("link") or "",
                "snippet": re.sub(r"<[^>]+>", " ", node.findtext("description") or ""),
                "provider": "bing_rss",
            }
        )
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
        from ddgs import DDGS
    except ImportError:
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
