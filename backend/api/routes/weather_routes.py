from fastapi import APIRouter, HTTPException, Query
from services.weather_api_service import weather_api_service
from services.ml_service import ml_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/current/{city}")
async def get_current_weather(city: str):
    """
    Obtiene datos meteorológicos actuales de una ciudad
    
    Ejemplo: /api/v1/weather/current/Bogotá,CO
    """
    try:
        weather_data = await weather_api_service.get_current_weather(city=city)
        return weather_data
    except Exception as e:
        logger.error(f"Error obteniendo clima: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/airport/{icao}")
async def get_airport_weather(icao: str):
    """
    Obtiene clima de aeropuerto por código ICAO
    
    Códigos disponibles:
    - SKBO: El Dorado (Bogotá)
    - SKCL: Alfonso Bonilla (Cali)
    - SKMR: Olaya Herrera (Medellín)
    - SKRG: José María Córdova (Rionegro)
    - SKCG: Rafael Núñez (Cartagena)
    - SKBQ: Palonegro (Bucaramanga)
    - SKPE: Matecaña (Pereira)
    """
    try:
        weather_data = await weather_api_service.get_airport_weather(icao)
        return weather_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo clima de aeropuerto: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/airport/{icao}/risk")
async def get_airport_risk(icao: str):
    """
    Obtiene clima Y predicción de riesgo para un aeropuerto
    """
    try:
        # Obtener clima actual
        weather_data = await weather_api_service.get_airport_weather(icao)
        
        # Preparar datos para predicción
        prediction_data = {
            "temperatura": weather_data["temperatura"],
            "humedad": weather_data["humedad"],
            "viento": weather_data["viento"],
            "visibilidad": weather_data["visibilidad"]
        }
        
        # Predecir riesgo
        risk_prediction = ml_service.predict_risk(prediction_data)
        
        return {
            "aeropuerto": weather_data["airport_name"],
            "icao": weather_data["icao_code"],
            "clima": weather_data,
            "prediccion": {
                "riesgo": risk_prediction["risk_level"],
                "confianza": risk_prediction["confidence"],
                "probabilidades": risk_prediction["probabilities"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error en predicción de aeropuerto: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/{city}")
async def get_weather_forecast(
    city: str,
    days: int = Query(default=3, ge=1, le=5, description="Días de pronóstico (1-5)")
):
    """
    Obtiene pronóstico meteorológico de 1-5 días
    """
    try:
        forecast = await weather_api_service.get_forecast(city, days)
        return forecast
    except Exception as e:
        logger.error(f"Error obteniendo pronóstico: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test")
async def test_endpoint():
    """Endpoint de prueba"""
    return {
        "status": "ok",
        "message": "Weather routes funcionando",
        "api_key_configured": bool(weather_api_service.api_key)
    } 