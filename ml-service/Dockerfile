FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    SIGNBRIDGE_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY scripts ./scripts
COPY service ./service
COPY models ./models
COPY model ./model

# A placeholder model directory is kept in source control, but production
# images must contain the real M4 artifact rather than silently shipping one.
RUN if [ ! -s model/bigru/bigru_best.pt ]; then \
      echo "ERROR: model/bigru/bigru_best.pt is required to build the runtime image" >&2; \
      exit 1; \
    fi

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
