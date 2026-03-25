# Backend Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Install system dependencies needed by Pillow / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer — copy requirements first)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application source
COPY app.py labels.py ./

# The model is mounted as a volume at runtime (see docker-compose.yml)
# MODEL_PATH env var can override the default location
ENV MODEL_PATH=/app/regular_spiking_model.pt

EXPOSE 1028

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:1028/api/health')" || exit 1

# 2 workers, 120 s timeout for slow first inference
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:1028", "--timeout", "120", "app:app"]
