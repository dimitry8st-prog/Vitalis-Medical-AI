from __future__ import annotations

import asyncio
from collections.abc import Callable
from urllib.parse import urljoin

import httpx


async def fetch_bounded(
    url: str,
    *,
    allowed: Callable[[str], bool],
    max_bytes: int,
    timeout: float,
    user_agent: str,
) -> tuple[bytes, str, str, str | None]:
    """Validate every redirect before connecting and limit decompressed body bytes."""
    current = url
    async with (
        asyncio.timeout(timeout),
        httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, headers={"User-Agent": user_agent}
        ) as client,
    ):
        for _ in range(4):
            if not allowed(current):
                raise ValueError("destination is outside source policy")
            async with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("redirect without destination")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(data) + len(chunk) > max_bytes:
                        raise ValueError("document exceeds configured size limit")
                    data.extend(chunk)
                return (
                    bytes(data),
                    response.headers.get("content-type", "").lower(),
                    current,
                    response.encoding,
                )
    raise ValueError("too many redirects")
