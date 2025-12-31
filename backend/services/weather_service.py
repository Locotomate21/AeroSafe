import logging
from .weather_api_service import weather_api_service


logger = logging.getLogger(__name__)

async def get_weather_data(city: str):
    """
    Obtiene datos meteorológicos reales desde OpenWeather
    """
    try:
        logger.info(f"Solicitando clima real para: {city}")
        data = await weather_api_service.get_current_weather(city=city)
        return data
    except Exception as e:
        logger.error(f"Error obteniendo clima: {str(e)}")
        # Si falla la API, devolver datos simulados de respaldo
        return {
            "city": city,
            "temperatura": 20.0,
            "humedad": 65,
            "viento": 15.0,
            "visibilidad": 8000,
            "message": f"Error con API real, datos simulados. Detalle: {str(e)}"
        }
