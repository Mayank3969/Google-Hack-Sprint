FROM python:3.13-slim

# INTEGRATION_WARNING (B11): v2 Keras serving image, not the extraction/training
# environment. Person A must pass check_isl_release after supplying evidence.
# Artifact smoke during build proves loading only; it is NOT accuracy evidence.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    SIGNBRIDGE_DEVICE=cpu \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=2

WORKDIR /app

COPY requirements-isl.txt ./requirements-isl.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-isl.txt

COPY scripts ./scripts
COPY service ./service
COPY models/*.keras models/*.json ./models/

# Validates both real artifacts, hashes, labels, shapes and finite inference.
# This intentionally does not claim Person A's release gate has passed.
RUN python -m scripts.evaluate_isl_models --smoke

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
