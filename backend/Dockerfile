# backend/Dockerfile
FROM python:3.13-slim

LABEL maintainer="AeroSafe Team"
LABEL version="1.0.0"
LABEL description="AeroSafe Weather Risk Prediction API"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd -m -u 1000 aerosafe && \
    mkdir -p /app && \
    chown -R aerosafe:aerosafe /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --chown=aerosafe:aerosafe requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código del backend
COPY --chown=aerosafe:aerosafe . /app/backend/

# Crear directorios necesarios
RUN mkdir -p /app/logs /app/ml/data/models && \
    chown -R aerosafe:aerosafe /app/logs /app/ml

USER aerosafe

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]