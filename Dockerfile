# =============================================================
# Socratica — Dockerfile
# Multi-stage build: deps → production image
# =============================================================

# ---- Stage 1: dependency builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools (needed for some native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install only runtime deps (skip torch/transformers on CPU-only edge)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi uvicorn[standard] python-multipart httpx \
        slowapi aiosqlite sqlalchemy python-dotenv \
        Pillow opencv-python-headless \
        datasets huggingface-hub \
        pyyaml

# ---- Stage 2: production image ----
FROM python:3.11-slim AS production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY backend/  ./backend/
COPY frontend/ ./frontend/
COPY run.py    .
COPY training_config.yaml .

# Create runtime directories
RUN mkdir -p uploads data/processed

# Non-root user for security
RUN useradd -m -u 1001 socratica && chown -R socratica:socratica /app
USER socratica

EXPOSE 8000

ENV APP_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
