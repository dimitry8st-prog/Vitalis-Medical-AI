from __future__ import annotations

import asyncio
import io
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

import httpx
from pypdf import PdfReader

from app.rag.source_policy import is_primary_international_source


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            value = re.sub(r"\s+", " ", data).strip()
            if value:
                self.parts.append(value)


def _html_to_text(raw: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(raw)
    return "\n".join(parser.parts)


def _pdf_to_text(raw: bytes, max_pages: int = 80) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages: list[str] = []
    for page in reader.pages[:max_pages]:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _query_terms(query: str) -> list[str]:
    stop = {
        "международные", "рекомендации", "актуальные", "лечение", "ведение",
        "после", "пациент", "guideline", "clinical", "recommendation",
    }
    terms = [
        token.lower()
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{4,}", query or "")
        if token.lower() not in stop
    ]
    return list(dict.fromkeys(terms))[:12]


def select_relevant_excerpt(text: str, query: str, max_chars: int = 6000) -> str:
    """Select evidence-like passages without treating retrieved text as instructions."""
    clean = re.sub(r"[ \t]+", " ", text or "")
    blocks = [block.strip() for block in re.split(r"\n{2,}|(?<=[.!?])\s+", clean) if block.strip()]
    terms = _query_terms(query)
    clinical_terms = (
        "recommend", "guideline", "dose", "dosage", "indication", "contraindication",
        "rehabilitation", "surgery", "treatment", "therapy", "evidence", "grade",
        "рекоменду", "доз", "показан", "противопоказ", "реабилитац", "хирург",
    )

    ranked: list[tuple[int, int, str]] = []
    for index, block in enumerate(blocks):
        low = block.lower()
        score = sum(3 for term in terms if term in low)
        score += sum(1 for term in clinical_terms if term in low)
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


async def enrich_primary_documents(
    hits: list[dict[str, Any]],
    query: str,
    max_documents: int = 3,
    max_bytes: int = 8_000_000,
) -> list[dict[str, Any]]:
    """Fetch allowlisted primary sources and attach bounded evidence excerpts."""
    candidates = [
        (index, hit)
        for index, hit in enumerate(hits)
        if is_primary_international_source(str(hit.get("url") or ""))
    ][:max_documents]

    async def fetch_one(index: int, hit: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        item = dict(hit)
        item["full_text_checked"] = False
        url = str(item.get("url") or "")
        try:
            async with httpx.AsyncClient(
                timeout=15,
                follow_redirects=True,
                headers={"User-Agent": "Vitalis-Medical-Evidence/1.0"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                raw = response.content
                if len(raw) > max_bytes:
                    raise ValueError("document exceeds configured size limit")
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" in content_type or url.lower().endswith(".pdf"):
                    text = await asyncio.to_thread(_pdf_to_text, raw)
                    item["retrieved_format"] = "pdf"
                else:
                    text = _html_to_text(response.text)
                    item["retrieved_format"] = "html"
                excerpt = select_relevant_excerpt(text, query)
                if excerpt:
                    item["evidence_excerpt"] = excerpt
                    item["full_text_checked"] = True
                    item["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            item["fetch_error"] = type(exc).__name__
        return index, item

    if not candidates:
        return hits
    updates = await asyncio.gather(*(fetch_one(index, hit) for index, hit in candidates))
    enriched = [dict(hit) for hit in hits]
    for index, item in updates:
        enriched[index] = item
    return enriched
