import asyncio

from starlette.responses import JSONResponse


class DocumentPrivacyMiddleware:
    """Reject oversized uploads before multipart spooling and prevent response caching."""

    def __init__(self, app, *, settings):
        self.app = app
        self.settings = settings
        self.upload_slots = asyncio.Semaphore(2)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith(
            "/documents"
        ):
            return await self.app(scope, receive, send)

        async def private_send(message):
            if message["type"] == "http.response.start":
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in {b"cache-control", b"referrer-policy"}
                ]
                message = {
                    **message,
                    "headers": headers
                    + [
                        (b"cache-control", b"no-store"),
                        (b"referrer-policy", b"no-referrer"),
                    ],
                }
            await send(message)

        if (
            scope["method"] == "POST"
            and scope["path"].rstrip("/") == "/documents/analyze"
        ):
            if self.upload_slots.locked():
                return await JSONResponse(
                    {"detail": "Обработчик PDF занят. Повторите позже."},
                    status_code=429,
                )(scope, receive, private_send)
            await self.upload_slots.acquire()
            try:
                return await self._upload(scope, receive, private_send)
            finally:
                self.upload_slots.release()
        return await self.app(scope, receive, private_send)

    async def _upload(self, scope, receive, private_send):
        limit = self.settings.document_max_file_bytes + 65_536  # multipart framing
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                return await JSONResponse(
                    {"detail": "Размер загрузки превышает лимит."}, status_code=413
                )(scope, receive, private_send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        return await self.app(scope, bounded_receive, private_send)
