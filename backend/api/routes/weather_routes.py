from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
import logging

from services.weather_service import get_weather_data
from models.schemas import WeatherResponse
from api.dependencies import validate_icao_code, validate_city_format

router = APIRouter()
logger = logging.getLogger(__name__)

# ==================== WEATHER ENDPOINTS ====================

@router.get("/current/{city}", response_model=WeatherResponse)
async def get_current_weather(
    city: str = Depends(validate_city_format)
):
    """
    Obtiene datos meteorológicos actuales de una ciudad
    
    Args:
        city: Ciudad en formato "Ciudad,Código_País" (ej: "Bogotá,CO")
    
    Returns:
        Datos meteorológicos actuales
    
    Ejemplo:
        GET /api/v1/weather/current/Bogotá,CO
    """
    try:
        weather_data = await get_weather_data(city=city)
        return weather_data
    except ValueError as e:
        logger.warning(f"Formato de ciudad inválido: {city}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo clima para {city}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"No se pudo obtener el clima: {str(e)}"
        )


@router.get("/airport/{icao}")
async def get_airport_weather(
    icao: str = Depends(validate_icao_code)
):
    """
    Obtiene clima actual de un aeropuerto por código ICAO
    
    Args:
        icao: Código ICAO de 4 letras (ej: SKBO, KJFK, EGLL)
    
    Returns:
        Datos meteorológicos del aeropuerto
    
    Ejemplos:
        - SKBO: El Dorado, Bogotá
        - KJFK: JFK, New York
        - EGLL: Heathrow, London
    """
    try:
        # Importar servicio (puede ser mock o real)
        from services.aviation_weather_service import get_airport_weather_data
        
        weather_data = await get_airport_weather_data(icao)
        return weather_data
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo clima de aeropuerto {icao}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo obtener clima del aeropuerto: {str(e)}"
        )


@router.get("/airport/{icao}/metar")
async def get_airport_metar(
    icao: str = Depends(validate_icao_code)
):
    """
    Obtiene reporte METAR de un aeropuerto
    
    METAR: Meteorological Aerodrome Report
    Formato estándar de reporte meteorológico para aviación
    
    Args:
        icao: Código ICAO del aeropuerto
        
    Returns:
        Reporte METAR raw y parseado
    """
    try:
        from services.metar_taf_service import get_metar_data
        
        metar_data = await get_metar_data(icao)
        return metar_data
        
    except Exception as e:
        logger.error(f"Error obteniendo METAR para {icao}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo obtener METAR: {str(e)}"
        )


@router.get("/airport/{icao}/taf")
async def get_airport_taf(
    icao: str = Depends(validate_icao_code)
):
    """
    Obtiene pronóstico TAF de un aeropuerto
    
    TAF: Terminal Aerodrome Forecast
    Pronóstico meteorológico específico para aeropuertos (hasta 30 horas)
    
    Args:
        icao: Código ICAO del aeropuerto
        
    Returns:
        Pronóstico TAF raw y parseado
    """
    try:
        from services.metar_taf_service import get_taf_data
        
        taf_data = await get_taf_data(icao)
        return taf_data
        
    except Exception as e:
        logger.error(f"Error obteniendo TAF para {icao}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo obtener TAF: {str(e)}"
        )


@router.get("/forecast/{city}")
async def get_weather_forecast(
    city: str = Depends(validate_city_format),
    days: int = Query(
        default=3, 
        ge=1, 
        le=5, 
        description="Días de pronóstico (1-5)"
    )
):
    """
    Obtiene pronóstico meteorológico de 1-5 días
    
    Args:
        city: Ciudad en formato "Ciudad,Código_País"
        days: Número de días (1-5)
        
    Returns:
        Pronóstico meteorológico
    """
    try:
        from services.weather_service import get_forecast_data
        
        forecast = await get_forecast_data(city, days)
        return forecast
        
    except Exception as e:
        logger.error(f"Error obteniendo pronóstico para {city}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo obtener pronóstico: {str(e)}"
        )


@router.get("/coordinates")
async def get_weather_by_coordinates(
    lat: float = Query(..., description="Latitud (-90 a 90)"),
    lon: float = Query(..., description="Longitud (-180 a 180)")
):
    """
    Obtiene clima por coordenadas geográficas
    
    Args:
        lat: Latitud
        lon: Longitud
        
    Returns:
        Datos meteorológicos de la ubicación
    """
    # Validar coordenadas
    if not -90 <= lat <= 90:
        raise HTTPException(
            status_code=400,
            detail="Latitud debe estar entre -90 y 90"
        )
    
    if not -180 <= lon <= 180:
        raise HTTPException(
            status_code=400,
            detail="Longitud debe estar entre -180 y 180"
        )
    
    try:
        from services.weather_service import get_weather_by_coords
        
        weather_data = await get_weather_by_coords(lat, lon)
        return weather_data
        
    except Exception as e:
        logger.error(f"Error obteniendo clima para ({lat}, {lon}): {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo obtener clima: {str(e)}"
        )


@router.get("/test")
async def test_weather_endpoint():
    """
    Endpoint de prueba para verificar que el servicio funciona
    """
    try:
        from core.config import settings
        
        return {
            "status": "ok",
            "message": "Weather routes funcionando correctamente",
            "api_configured": bool(settings.OPENWEATHER_API_KEY),
            "endpoints": {
                "current": "/api/v1/weather/current/{city}",
                "airport": "/api/v1/weather/airport/{icao}",
                "metar": "/api/v1/weather/airport/{icao}/metar",
                "taf": "/api/v1/weather/airport/{icao}/taf",
                "forecast": "/api/v1/weather/forecast/{city}",
                "coordinates": "/api/v1/weather/coordinates?lat={lat}&lon={lon}"
            }
        }
    except Exception as e:
        logger.error(f"Error en test endpoint: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }