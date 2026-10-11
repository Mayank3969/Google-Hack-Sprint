"""SignBridge backend application entry point.

Module BE-M1 provides the service skeleton and health check. The process route
is implemented by BE-M4, the transcribe route by BE-M6, and the session routes
by BE-M8.
"""

from fastapi import FastAPI

if __package__:
    from .routes import process, session, transcribe
else:  # Supports ``uvicorn main:app`` from the backend directory.
    from routes import process, session, transcribe


app = FastAPI(
    title="SignBridge Backend",
    version="0.1.0",
    description="Backend integration service for the SignBridge demo.",
)

app.include_router(process.router)
app.include_router(transcribe.router)
app.include_router(session.router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a cheap liveness response without calling any external service."""

    return {"status": "ok"}
