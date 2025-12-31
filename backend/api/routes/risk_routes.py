from fastapi import APIRouter, HTTPException
from backend.models.schemas import RiskRequest, RiskResponse
from backend.services.ml_service import ml_service
from backend.services.weather_service import get_weather_data
import logging
from backend.services.weather_service import get_weather_data
from backend.services.risk_predictor import RiskPredictor
from backend.models.schemas import RiskRequest, RiskResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=RiskResponse)
async def predict_risk(request: RiskRequest):
    """
    Predice el nivel de riesgo basado en condiciones meteorológicas
    """
    try:
        logger.info(f"Predicción solicitada: {request.dict()}")
        
        # Preparar datos para predicción
        weather_data = {
            "temperatura": request.temperatura,
            "humedad": request.humedad,
            "viento": request.viento,
            "visibilidad": request.visibilidad
        }
        
        # Predecir riesgo
        prediction = ml_service.predict_risk(weather_data)
        
        return RiskResponse(
            riesgo=prediction["risk_level"],
            confianza=prediction["confidence"],
            probabilidades=prediction["probabilities"],
            datos_clima=weather_data
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al predecir riesgo: {str(e)}")


@router.get("/test")
async def test_endpoint():
    """Endpoint de prueba"""
    return {"status": "ok", "message": "Risk routes funcionando"}
