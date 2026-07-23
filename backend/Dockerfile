# AeroSafe - Imagen de la API
#
# Construir desde backend/:   docker build -t aerosafe-api .
#
# WORKDIR es /app y ahi se copia el contenido de backend/, de modo que
# los imports quedan igual que en local (from core.config import ...).

FROM python:3.13-slim

LABEL maintainer="AeroSafe Team"
LABEL description="AeroSafe Weather Risk Prediction API"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl es necesario para el HEALTHCHECK; gcc/g++ para compilar las
# dependencias cientificas que no traen rueda para esta plataforma.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 aerosafe

WORKDIR /app

# Las dependencias van primero y en su propia capa: cambiar el codigo no
# invalida la instalacion de paquetes.
# Solo requirements.txt (runtime). Las de entrenamiento y test no entran
# en la imagen que se despliega.
COPY --chown=aerosafe:aerosafe requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=aerosafe:aerosafe . /app/

RUN mkdir -p /app/logs /app/data/batch && \
    chown -R aerosafe:aerosafe /app/logs /app/data

USER aerosafe

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
