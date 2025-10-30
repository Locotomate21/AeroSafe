import logging

logger = logging.getLogger(__name__)

async def get_weather_data(city: str):
    """
    Obtiene datos meteorológicos (placeholder)
    TODO: Implementar integración con API real
    """
    logger.info(f"Solicitando datos para: {city}")
    
    # Datos de ejemplo
    return {
        "city": city,
        "temperatura": 20.0,
        "humedad": 65,
        "viento": 15.0,
        "visibilidad": 8000,
        "message": "Datos de ejemplo - implementar API real"
    }
