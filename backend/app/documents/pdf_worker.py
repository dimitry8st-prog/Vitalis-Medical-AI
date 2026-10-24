"""Credential-free PDF parser process. Never execute PDF actions or open its links."""

from __future__ import annotations

import base64
import json
import sys
from dataclasses import asdict


def main() -> None:
    # Limits apply before importing the PDF library. Docker adds an outer memory cap.
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (192 * 1024**2, 192 * 1024**2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
    except ImportError:  # Windows: wall timeout still terminates the process.
        pass

    from app.documents.extractor import DocumentValidationError, extract_pdf

    try:
        payload = json.loads(sys.stdin.buffer.read(12_000_000))
        data = base64.b64decode(payload["data"], validate=True)
        result = extract_pdf(
            data, max_pages=payload["max_pages"], max_chars=payload["max_chars"]
        )
        output = {"result": asdict(result)}
    except DocumentValidationError as exc:
        output = {"error": str(exc)}
    except Exception:
        output = {"error": "Не удалось безопасно обработать PDF."}
    sys.stdout.write(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
