"""Import the approved Russian ICHD-3 PDF into the local Vitalis vault."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.documents.isolated_parser import extract_pdf_isolated
from app.rag.eligibility import ICHD3_SCOPE, ICHD3_SHA256, ICHD3_URL

TITLE = "МКГБ-3 — Международная классификация головной боли"
OUTPUT = Path("generated/classifications/intl/ichd3-ru.md")


async def import_ichd3(pdf: Path, vault: Path) -> tuple[Path, int]:
    data = pdf.read_bytes()
    if hashlib.sha256(data).hexdigest() != ICHD3_SHA256:
        raise ValueError("SHA-256 не совпадает с официальной русской версией МКГБ-3.")
    document = await extract_pdf_isolated(
        data, max_pages=206, max_chars=900_000, timeout=60,
    )
    if len(document.pages) != 206:
        raise ValueError("Не удалось извлечь полный текст МКГБ-3.")

    metadata = {
        "title": TITLE,
        "year": 2018,
        "source": "International Headache Society",
        "source_url": ICHD3_URL,
        "source_version": "ICHD-3 (2018), Russian translation",
        "jurisdiction": "INTL",
        "language": "ru",
        "document_type": "diagnostic_classification",
        "content_scope": ICHD3_SCOPE,
        "full_text_checked": True,
        "full_text_sha256": document.sha256,
        "page_count": len(document.pages),
        "specialty_tags": "neurology",
    }
    header = "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items())
    body = "\n\n".join(f"## Страница {page.number}\n\n{page.text}" for page in document.pages)
    dest = vault / OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=dest.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(f"---\n{header}\n---\n\n{body}\n")
    try:
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    return dest, len(document.pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--vault", type=Path, required=True)
    args = parser.parse_args()
    dest, pages = asyncio.run(import_ichd3(args.pdf, args.vault))
    print(f"{dest}: {pages} pages")


if __name__ == "__main__":
    main()
