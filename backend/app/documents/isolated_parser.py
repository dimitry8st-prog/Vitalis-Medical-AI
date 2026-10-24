from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

from app.documents.extractor import DocumentValidationError
from app.documents.models import DocumentPage, ExtractedDocument

_slots = asyncio.Semaphore(1)


async def extract_pdf_isolated(
    data: bytes,
    *,
    max_pages: int,
    max_chars: int,
    timeout: float = 20,
) -> ExtractedDocument:
    """Bound parser lifetime, concurrency and memory; do not inherit API secrets."""
    async with _slots:
        backend = str(Path(__file__).resolve().parents[2])
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "app.documents.pdf_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=backend,
            env={
                "PYTHONPATH": backend,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                **(
                    {"SYSTEMROOT": os.environ["SYSTEMROOT"]}
                    if "SYSTEMROOT" in os.environ
                    else {}
                ),
            },
        )
        payload = json.dumps(
            {
                "data": base64.b64encode(data).decode("ascii"),
                "max_pages": max_pages,
                "max_chars": max_chars,
            }
        ).encode()
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(payload), timeout=timeout
            )
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        try:
            output = json.loads(stdout)
            if "error" in output:
                raise DocumentValidationError(output["error"])
            result = output["result"]
            result["pages"] = tuple(DocumentPage(**page) for page in result["pages"])
            return ExtractedDocument(**result)
        except (ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, DocumentValidationError):
                raise
            raise DocumentValidationError(
                "Изолированный PDF-парсер завершился без результата."
            ) from exc
