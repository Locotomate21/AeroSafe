"""
Endpoints de pronóstico.

Sirven el modelo de pronóstico calibrado: predicen la probabilidad de
niebla o tormenta a N horas a partir del METAR actual del aeropuerto.

Es distinto de /risk, que clasifica la condición actual. Aquí se predice
el futuro, con probabilidad calibrada apta para decidir.
"""
import logging

from fastapi import APIRouter, HTTPException, Path

from models.schemas import ForecastResponse
from services.forecast_service import (
    HORIZONTE_H,
    MetarIncompleto,
    MetarNoDisponible,
    get_forecast_service,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Aeropuertos con modelo de pronóstico entrenado y validado.
AEROPUERTOS_SOPORTADOS = {"SKBO", "SKRG", "SKPS", "SKMZ"}


@router.get("/{icao}", response_model=ForecastResponse)
async def pronostico_aeropuerto(
    icao: str = Path(..., min_length=4, max_length=4, description="Código ICAO"),
):
    """
    Pronostica niebla o tormenta a 3 horas para un aeropuerto.

    Descarga el METAR actual del aeropuerto, lo pasa por el pipeline de
    features y el modelo calibrado, y devuelve la probabilidad.

    La probabilidad está **calibrada**: 0.30 significa ~30% de ocurrencia
    real, no un score sin escala. El campo `nivel` la traduce a
    MINIMO/BAJO/MODERADO/ALTO para lectura rápida.

    Aeropuertos soportados: SKBO, SKRG, SKPS, SKMZ.
    """
    icao = icao.upper().strip()

    if not icao.isalpha():
        raise HTTPException(status_code=400, detail="Código ICAO debe ser alfabético")

    if icao not in AEROPUERTOS_SOPORTADOS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No hay modelo de pronóstico para {icao}. "
                f"Soportados: {', '.join(sorted(AEROPUERTOS_SOPORTADOS))}."
            ),
        )

    servicio = get_forecast_service(icao, HORIZONTE_H)
    if not servicio.disponible():
        raise HTTPException(
            status_code=503,
            detail=f"Modelo de pronóstico de {icao} no cargado.",
        )

    try:
        return await servicio.pronosticar(icao)
    except MetarNoDisponible as e:
        # La fuente externa (NOAA) no respondió: no es culpa del cliente.
        raise HTTPException(status_code=503, detail=str(e))
    except MetarIncompleto as e:
        # El METAR llegó pero no sirve para pronosticar.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Error en pronóstico de %s: %s", icao, e, exc_info=True)
        raise HTTPException(status_code=502, detail="Error al obtener el pronóstico")


@router.get("/", include_in_schema=False)
async def listar_soportados():
    """Lista los aeropuertos con pronóstico disponible."""
    return {
        "aeropuertos_soportados": sorted(AEROPUERTOS_SOPORTADOS),
        "horizonte_horas": HORIZONTE_H,
        "objetivo": "niebla o tormenta",
    }
