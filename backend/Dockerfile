FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MOCK_EXTERNAL_SERVICES=0

WORKDIR /app

# Install dependencies in a separate layer so source-only changes rebuild quickly.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Keep the package rooted at /app so both package-relative imports and the
# Cloud Run entry point (backend.main:app) resolve consistently.
COPY . ./backend

EXPOSE 8080

CMD ["sh", "-c", "exec python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
