from __future__ import annotations

import asyncio
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

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
    en_focus = (
        await _resolve_english_focus(focus)
        if scope in ("intl", "both")
        else _english_variant(focus)
    )

    literature_task = (
        asyncio.create_task(_literature_bundle(query, max_results, en_focus))
        if scope in ("intl", "both")
        else None
    )

    queries = _build_queries(
        query,
        bilingual=bilingual,
        freshness_extra=freshness_extra,
        scope=scope,
        en_focus=en_focus,
    )
    semaphore = asyncio.Semaphore(6 if scope == "rf" else 5)

    async def run_one(item: tuple[str, str, str]) -> tuple[list[dict[str, Any]], str, str]:
        q, lang, kind = item
        async with semaphore:
            try:
                batch = await asyncio.wait_for(
                    (
                        _run_rf_provider(provider, q, settings, max_results)
                        if lang == "ru"
                        else _run_provider(provider, q, settings, max_results)
                    ),
                    timeout=12.0,
                )
            except Exception:
                batch = []
        return batch, lang, kind

    batches = await asyncio.gather(*(run_one(item) for item in queries))
    for batch, lang, kind in batches:
        _merge(all_hits, seen, batch, lang, kind)

    if literature_task is not None:
        for batch, lang, kind in await literature_task:
            _merge(all_hits, seen, batch, lang, kind)

    for item in all_hits:
        item["relevance_score"] = _score_hit(item, f"{focus} {en_focus}")
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
        if lang == "ru":
            item["source_tier"] = _rf_source_tier(str(item.get("url") or ""))
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
    en_focus: str | None = None,
) -> list[tuple[str, str, str]]:
    year = datetime.now().year
    prev = year - 1
    focus = _extract_focus(query)
    ru_focus = _canonical_ru_focus(focus)
    en_focus = en_focus or _english_variant(focus)
    intent_ru, intent_en = _detect_intent(query)
    out: list[tuple[str, str, str]] = []

    if scope in ("rf", "both"):
        out.extend(
            [
                (f'site:cr.minzdrav.gov.ru {ru_focus} {intent_ru}', "ru", "rf_registry"),
                (f'{ru_focus} {intent_ru} клинические рекомендации РФ Минздрав', "ru", "rf_guideline"),
                (f'{ru_focus} клинические рекомендации РФ filetype:pdf', "ru", "rf_guideline_pdf"),
                (f'{ru_focus} {intent_ru} НМИЦ профессиональное общество рекомендации', "ru", "rf_institute"),
                (f'{ru_focus} {intent_ru} клинический обзор медицинский журнал', "ru", "rf_journal"),
            ]
        )
        if freshness_extra:
            out.append(
                (f'{ru_focus} клинические рекомендации {prev} OR {year}', "ru", "rf_freshness")
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


async def _run_rf_provider(
    provider: str,
    query: str,
    settings: Settings,
    max_results: int,
) -> list[dict[str, Any]]:
    """RF-only fallback chain; the international contour stays unchanged."""
    tasks = {
        asyncio.create_task(_run_provider(provider, query, settings, max_results)),
        asyncio.create_task(_duckduckgo_html(query, max_results)),
        asyncio.create_task(_brave_html(query, max_results)),
    }
    done, pending = await asyncio.wait(tasks, timeout=11.0)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in done:
        try:
            batch = task.result()
        except Exception:
            continue
        for item in batch:
            key = str(item.get("url") or "").strip()
            if key and key not in seen:
                seen.add(key)
                rows.append(item)
    return rows


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


_RU_EN_PATTERNS = (
    (r"ишемическ\w*\s+инсульт\w*", "ischemic stroke"),
    (r"геморрагическ\w*\s+инсульт\w*", "hemorrhagic stroke"),
    (r"черепно[- ]мозгов\w*\s+травм\w*", "traumatic brain injury"),
    (r"рассеянн\w*\s+склероз\w*", "multiple sclerosis"),
    (r"болезн\w*\s+паркинсон\w*", "Parkinson disease"),
    (r"болезн\w*\s+альцгеймер\w*", "Alzheimer disease"),
    (r"сердечн\w*\s+недостаточност\w*", "heart failure"),
    (r"фибрилляц\w*\s+предсерди\w*", "atrial fibrillation"),
    (r"хроническ\w*\s+болезн\w*\s+почек", "chronic kidney disease"),
    (r"рак\w*\s+молочн\w*\s+желез\w*", "breast cancer"),
    (r"рак\w*\s+предстательн\w*\s+желез\w*", "prostate cancer"),
)

_RU_CANONICAL_PATTERNS = (
    (r"ишемическ\w*\s+инсульт\w*", "ишемический инсульт"),
    (r"геморрагическ\w*\s+инсульт\w*", "геморрагический инсульт"),
    (r"черепно[- ]мозгов\w*\s+травм\w*", "черепно-мозговая травма"),
    (r"рассеянн\w*\s+склероз\w*", "рассеянный склероз"),
    (r"болезн\w*\s+паркинсон\w*", "болезнь Паркинсона"),
    (r"болезн\w*\s+альцгеймер\w*", "болезнь Альцгеймера"),
    (r"сердечн\w*\s+недостаточност\w*", "сердечная недостаточность"),
    (r"фибрилляц\w*\s+предсерди\w*", "фибрилляция предсердий"),
    (r"хроническ\w*\s+болезн\w*\s+почек", "хроническая болезнь почек"),
    (r"рак\w*\s+молочн\w*\s+желез\w*", "рак молочной железы"),
    (r"рак\w*\s+предстательн\w*\s+желез\w*", "рак предстательной железы"),
)

_INTENT_TERMS = (
    (r"леч|терап|препарат|дозиров", "лечение", "treatment"),
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


def _canonical_ru_focus(focus: str) -> str:
    """Normalize frequent inflected diagnoses; otherwise keep unquoted user keywords."""
    low = (focus or "").lower()
    for pattern, canonical in _RU_CANONICAL_PATTERNS:
        if re.search(pattern, low):
            return canonical
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё. -]+", " ", focus).strip()


def _english_variant(query: str) -> str:
    low = (query or "").lower()
    matched: list[str] = [
        en for pattern, en in _RU_EN_PATTERNS if re.search(pattern, low)
    ]
    for ru, en in sorted(_RU_EN_TERMS.items(), key=lambda pair: len(pair[0]), reverse=True):
        if ru in low and en not in matched:
            matched.append(en)
    codes = re.findall(r"\b[A-Z]\d{2}(?:\.\d+)?\b", query or "", flags=re.I)
    if matched:
        specific = [
            term for term in matched
            if not any(term != other and term.lower() in other.lower() for other in matched)
        ]
        return " ".join(specific + codes)
    latin = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query or "")
    return " ".join(latin + codes) or _extract_focus(query)


async def _resolve_english_focus(focus: str) -> str:
    """Use curated terminology first; Wikipedia only translates a search concept."""
    mapped = _english_variant(focus)
    if mapped != focus and not re.search(r"[А-Яа-яЁё]", mapped):
        return mapped
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(
                "https://ru.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": focus,
                    "gsrlimit": 1,
                    "prop": "langlinks",
                    "lllang": "en",
                    "lllimit": 1,
                    "format": "json",
                },
                headers={"User-Agent": "Vitalis/1.0"},
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            for page in pages.values():
                links = page.get("langlinks") or []
                if links and links[0].get("*"):
                    return str(links[0]["*"])
    except Exception:
        pass
    return mapped

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
    query: str, max_results: int, en_focus: str | None = None
) -> list[tuple[list[dict[str, Any]], str, str]]:
    focus = en_focus or _english_variant(_extract_focus(query))
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
    overlap = sum(
        1
        for word in words
        if word in haystack
        or (
            re.search(r"[а-яё]", word)
            and len(word) >= 7
            and word[:6] in haystack
        )
    )
    authority = 0
    tiers = (
        (5, ("cr.minzdrav.gov.ru", "minzdrav.gov.ru", "who.int", "nice.org.uk")),
        (4, ("nih.gov", "cdc.gov", "pubmed.ncbi.nlm.nih.gov", "europepmc.org", "cochranelibrary.com", "nejm.org", "thelancet.com", "jamanetwork.com", "bmj.com", "nature.com")),
        (3, ("escardio.org", "heart.org", "acc.org", "idsociety.org", "nccn.org", "eular.org", "ersnet.org")),
        (2, ("sechenov.ru", "rsmu.ru", "nmicr.ru", "nmicrk.ru", "oncology.ru", "almazovcentre.ru", "gnicpm.ru", "mediasphera.ru", "journals.eco-vector.com", "rnmot.ru", "neurology.ru", "elibrary.ru", "elpub.ru", "orscience.ru", "ter-arkhiv.ru", "rehabrus.ru", "medpoint.pro", "diseases.medelement.com", "evidence-neurology.ru", "consultant.ru", "garant.ru")),
    )
    for points, domains in tiers:
        if any(domain in url for domain in domains):
            authority = points
            break
    year = item.get("year") or 0
    freshness = 2 if year >= datetime.now().year - 2 else 1 if year >= datetime.now().year - 5 else 0
    mismatch = 0
    if "ишемич" in focus.lower() and "геморраг" in haystack:
        mismatch = 8
    elif "геморраг" in focus.lower() and "ишемич" in haystack:
        mismatch = 8
    return float(overlap * 2 + authority + freshness - mismatch)


def _balanced_results(
    hits: list[dict[str, Any]], scope: str, max_results: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        hits,
        key=lambda item: (item.get("relevance_score", 0), item.get("year") or 0),
        reverse=True,
    )
    if scope == "rf":
        return [
            item
            for item in ranked
            if item.get("lang") == "ru" and item.get("relevance_score", 0) > 0
        ][: max_results * 2]
    if scope == "intl":
        return [item for item in ranked if item.get("lang") == "en"][: max_results * 2]
    ru = [
        item
        for item in ranked
        if item.get("lang") == "ru" and item.get("relevance_score", 0) > 0
    ][:max_results]
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
                params={
                    "q": query,
                    "format": "rss",
                    "setlang": "ru" if re.search(r"[А-Яа-яЁё]", query) else "en",
                },
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


async def _duckduckgo_html(query: str, max_results: int) -> list[dict[str, Any]]:
    """Independent RF fallback when the DDGS package and Bing RSS return no rows."""
    try:
        async with httpx.AsyncClient(
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Vitalis/1.0",
                "Accept-Language": "ru-RU,ru;q=0.9",
            },
            follow_redirects=True,
        ) as client:
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query, "kl": "ru-ru"},
            )
            response.raise_for_status()
            body = response.text
    except Exception:
        return []

    anchors = re.findall(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        body,
        flags=re.I | re.S,
    )
    snippets = re.findall(
        r'<(?:a|div)[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
        body,
        flags=re.I | re.S,
    )
    out: list[dict[str, Any]] = []
    for index, (raw_url, raw_title) in enumerate(anchors[:max_results]):
        out.append(
            {
                "title": _strip_html(raw_title),
                "url": _decode_ddg_url(html.unescape(raw_url)),
                "snippet": _strip_html(snippets[index]) if index < len(snippets) else "",
                "provider": "duckduckgo_html",
            }
        )
    return [row for row in out if row["url"]]


async def _brave_html(query: str, max_results: int) -> list[dict[str, Any]]:
    """Public HTML fallback for RF discovery; no API key and no answer generation."""
    try:
        async with httpx.AsyncClient(
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Language": "ru-RU,ru;q=0.9",
            },
            follow_redirects=True,
        ) as client:
            response = await client.get(
                "https://search.brave.com/search",
                params={"q": query, "source": "web"},
            )
            response.raise_for_status()
            body = response.text
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    blocks = re.split(
        r'<div class="snippet[^\"]*"[^>]+data-type="web"[^>]*>',
        body,
        flags=re.I,
    )[1:]
    for block in blocks:
        match = re.search(
            r'<a href="(https?://[^\"]+)"[^>]*class="[^\"]*\bl1\b[^\"]*"[^>]*>'
            r'.*?<div class="title search-snippet-title[^\"]*" title="([^\"]+)"',
            block,
            flags=re.I | re.S,
        )
        if not match:
            continue
        snippet_match = re.search(
            r'<div class="generic-snippet[^\"]*">.*?<div class="content[^\"]*">(.*?)</div>',
            block,
            flags=re.I | re.S,
        )
        out.append(
            {
                "title": html.unescape(match.group(2)).strip(),
                "url": html.unescape(match.group(1)).strip(),
                "snippet": _strip_html(snippet_match.group(1)) if snippet_match else "",
                "provider": "brave_html",
            }
        )
        if len(out) >= max_results:
            break
    return out


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def _decode_ddg_url(value: str) -> str:
    if value.startswith("//"):
        value = f"https:{value}"
    parsed = urlparse(value)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target) if target else ""
    return value


def _rf_source_tier(url: str) -> str:
    host = (urlparse(url or "").hostname or "").lower()
    matches = lambda domain: host == domain or host.endswith(f".{domain}")
    if any(matches(domain) for domain in ("cr.minzdrav.gov.ru", "minzdrav.gov.ru")):
        return "official"
    if any(matches(domain) for domain in ("sechenov.ru", "rsmu.ru", "nmicr.ru", "nmicrk.ru", "almazovcentre.ru", "gnicpm.ru", "rnmot.ru", "oncology.ru", "neurology.ru")):
        return "institute_or_society"
    if any(matches(domain) for domain in ("mediasphera.ru", "journals.eco-vector.com", "elibrary.ru", "elpub.ru", "orscience.ru", "ter-arkhiv.ru")):
        return "peer_reviewed_literature"
    if any(matches(domain) for domain in ("rehabrus.ru", "medpoint.pro", "diseases.medelement.com", "evidence-neurology.ru", "consultant.ru", "garant.ru")):
        return "trusted_mirror"
    return "other"


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
