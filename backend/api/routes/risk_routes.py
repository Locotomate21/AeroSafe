from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging

from models.schemas import RiskRequest, RiskResponse
from api.dependencies import get_db
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)

# ==================== RISK PREDICTION ENDPOINTS ====================

@router.post("/predict", response_model=RiskResponse)
async def predict_risk(request: RiskRequest):
    """
    Predice el nivel de riesgo meteorológico basado en condiciones actuales
    
    Args:
        request: Datos meteorológicos para análisis
        
    Returns:
        Predicción de riesgo con nivel, confianza y probabilidades
        
    Niveles de riesgo:
        - BAJO: Condiciones normales, operaciones seguras
        - MODERADO: Precaución recomendada, monitoreo
        - ALTO: Condiciones adversas, posibles restricciones
        - CRÍTICO: Peligro severo, operaciones limitadas
    
    Ejemplo de request:
        {
            "temperatura": 15.5,
            "humedad": 65,
            "viento": 12,
            "visibilidad": 9999,
            "presion": 1013,
            "condicion": "Nublado"
        }
    """
    try:
        logger.info(f"Predicción de riesgo solicitada: {request.dict()}")
        
        # Importar servicio ML
        from services.ml_service_v2 import predict_risk_from_weather
        
        # Preparar datos para predicción
        weather_data = {
            "temperatura": request.temperatura,
            "humedad": request.humedad,
            "viento": request.viento,
            "visibilidad": request.visibilidad,
            "presion": getattr(request, 'presion', 1013),
            "condicion": getattr(request, 'condicion', 'Unknown')
        }
        
        # Predecir riesgo
        prediction = await predict_risk_from_weather(weather_data)
        
        response = RiskResponse(
            riesgo=prediction["risk_level"],
            confianza=prediction["confidence"],
            probabilidades=prediction.get("probabilities", {}),
            factores_riesgo=prediction.get("risk_factors", []),
            recomendaciones=prediction.get("recommendations", []),
            datos_clima=weather_data
        )
        
        logger.info(f"Predicción completada: {prediction['risk_level']}")
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción de riesgo: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al predecir riesgo: {str(e)}"
        )


@router.post("/predict/airport/{icao}")
async def predict_airport_risk(icao: str):
    """
    Predice riesgo para un aeropuerto específico usando clima actual
    
    Args:
        icao: Código ICAO del aeropuerto (ej: SKBO, KJFK)
        
    Returns:
        Clima actual + predicción de riesgo
    """
    try:
        logger.info(f"Predicción de riesgo para aeropuerto: {icao}")
        
        # Validar ICAO
        icao = icao.upper().strip()
        if len(icao) != 4 or not icao.isalpha():
            raise HTTPException(
                status_code=400,
                detail="Código ICAO debe tener 4 letras"
            )
        
        # Obtener clima del aeropuerto
        from services.aviation_weather_service import get_airport_weather_data
        weather_data = await get_airport_weather_data(icao)
        
        # Predecir riesgo
        from services.ml_service_v2 import predict_risk_from_weather
        prediction = await predict_risk_from_weather(weather_data)
        
        return {
            "aeropuerto": {
                "icao": icao,
                "nombre": weather_data.get("airport_name", "Unknown"),
            },
            "clima_actual": weather_data,
            "prediccion_riesgo": {
                "nivel": prediction["risk_level"],
                "confianza": prediction["confidence"],
                "probabilidades": prediction.get("probabilities", {}),
                "factores": prediction.get("risk_factors", []),
                "recomendaciones": prediction.get("recommendations", [])
            },
            "timestamp": weather_data.get("timestamp")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción para aeropuerto {icao}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener predicción: {str(e)}"
        )


@router.get("/history")
async def get_risk_history(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Obtiene historial de predicciones de riesgo
    
    Args:
        limit: Número máximo de registros a retornar
        db: Sesión de base de datos
        
    Returns:
        Lista de predicciones históricas
    """
    try:
        # TODO: Implementar consulta a base de datos
        # Por ahora retorna datos de ejemplo
        
        return {
            "total": 0,
            "predictions": [],
            "message": "Historial no implementado aún"
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo historial: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener historial: {str(e)}"
        )


@router.get("/stats")
async def get_risk_statistics(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Obtiene estadísticas de predicciones de riesgo
    
    Args:
        days: Número de días a analizar
        db: Sesión de base de datos
        
    Returns:
        Estadísticas agregadas de riesgo
    """
    try:
        # TODO: Implementar análisis estadístico
        
        return {
            "period_days": days,
            "total_predictions": 0,
            "risk_distribution": {
                "BAJO": 0,
                "MODERADO": 0,
                "ALTO": 0,
                "CRÍTICO": 0
            },
            "average_confidence": 0.0,
            "message": "Estadísticas no implementadas aún"
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener estadísticas: {str(e)}"
        )


@router.get("/test")
async def test_risk_endpoint():
    """
    Endpoint de prueba para verificar que el servicio funciona
    """
    try:
        from core.config import settings
        from pathlib import Path
        
        # Verificar que el modelo ML existe
        model_path = settings.get_model_path(settings.MODEL_PATH)
        model_exists = model_path.exists()
        
        return {
            "status": "ok",
            "message": "Risk routes funcionando correctamente",
            "ml_model": {
                "configured": bool(settings.MODEL_PATH),
                "exists": model_exists,
                "path": str(model_path)
            },
            "endpoints": {
                "predict": "POST /api/v1/risk/predict",
                "predict_airport": "POST /api/v1/risk/predict/airport/{icao}",
                "history": "GET /api/v1/risk/history",
                "stats": "GET /api/v1/risk/stats"
            }
        }
        
    except Exception as e:
        logger.error(f"Error en test endpoint: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/demo")
async def demo_prediction():
    """
    Endpoint de demostración con datos de ejemplo
    
    Útil para probar la API sin necesidad de datos reales
    """
    # Datos de ejemplo: condiciones típicas de un día en Bogotá
    demo_request = RiskRequest(
        temperatura=18.5,
        humedad=70,
        viento=8,
        visibilidad=9000,
        presion=756,  # mmHg a 2600m altitud
        condicion="Parcialmente nublado"
    )
    
    try:
        # Usar el endpoint de predicción
        return await predict_risk(demo_request)
        
    except Exception as e:
        logger.error(f"Error en demo: {str(e)}")
        # Si falla, retornar datos mock
        return {
            "riesgo": "BAJO",
            "confianza": 0.85,
            "probabilidades": {
                "BAJO": 0.85,
                "MODERADO": 0.12,
                "ALTO": 0.03,
                "CRÍTICO": 0.00
            },
            "factores_riesgo": ["Visibilidad reducida"],
            "recomendaciones": ["Monitoreo continuo recomendado"],
            "datos_clima": demo_request.dict(),
            "note": "Datos de demostración - servicio ML no disponible"
        }