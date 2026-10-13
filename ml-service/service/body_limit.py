"""Bound HTTP bodies before JSON parsing, including chunked uploads.

INTEGRATION_WARNING (B5/B8): Person A changes feature schemas, not these byte
limits. Re-measure the maximum valid batch before changing limits at integration.
"""

from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self, app, default_limit: int, batch_limit: int):
        self.app = app
        self.default_limit = default_limit
        self.batch_limit = batch_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.batch_limit if scope["path"].rstrip("/") in {
            "/v2/predict", "/v2/predict/batch"
        } else self.default_limit
        headers = dict(scope.get("headers", []))
        length = headers.get(b"content-length")
        if length is not None:
            try:
                length = int(length)
                if length < 0:
                    raise ValueError
            except ValueError:
                return await JSONResponse({"detail": "invalid content-length header"}, 400)(scope, receive, send)
            if length > limit:
                return await JSONResponse({"detail": "request body is too large"}, 413)(scope, receive, send)
        parts = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            part = message.get("body", b"")
            size += len(part)
            if size > limit:
                return await JSONResponse({"detail": "request body is too large"}, 413)(scope, receive, send)
            parts.append(part)
            if not message.get("more_body", False):
                break
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"".join(parts), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
