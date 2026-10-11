FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_PATH=/app/models/classifier.pkl

WORKDIR /app

# Runtime libraries required by OpenCV/MediaPipe in the slim base image.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in a separate layer so source-only changes rebuild quickly.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY service ./service
COPY scripts ./scripts
COPY models ./models

# The image does not require root to serve the API.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn service.app:app --host 0.0.0.0 --port ${PORT}"]
