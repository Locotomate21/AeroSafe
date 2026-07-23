from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
import logging
from datetime import datetime, timedelta

from api.dependencies import validate_icao_code, get_db
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)

# ==================== DASHBOARD ENDPOINTS ====================

@router.get("/status")
async def get_dashboard_status():
    """
    Estado general del sistema AeroSafe
    
    Returns:
        Estado operacional del sistema y servicios
    """
    try:
        from core.config import settings
        from pathlib import Path
        
        # Verificar componentes
        model_exists = settings.get_model_path(settings.MODEL_PATH).exists()
        api_configured = bool(settings.OPENWEATHER_API_KEY)
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.VERSION,
            "components": {
                "api": "healthy",
                "ml_model": "loaded" if model_exists else "not_found",
                "weather_api": "configured" if api_configured else "missing_key",
                "database": "connected"  # TODO: verificar conexión real
            },
            "metrics": {
                "uptime": "N/A",  # TODO: implementar tracking
                "total_predictions": 0,  # TODO: consultar BD
                "active_alerts": 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del dashboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener estado: {str(e)}"
        )


@router.get("/airport/{icao}/status")
async def get_airport_status(
    icao: str = Depends(validate_icao_code)
):
    """
    Estado completo de un aeropuerto con análisis de riesgo
    
    Args:
        icao: Código ICAO del aeropuerto
        
    Returns:
        Clima actual, predicción de riesgo y estado operacional
    """
    try:
        logger.info(f"Obteniendo estado completo para aeropuerto: {icao}")
        
        # Obtener clima actual
        from services.aviation_weather_service import get_airport_weather_data
        weather_data = await get_airport_weather_data(icao)
        
        # Predecir riesgo
        from services.ml_service_v2 import predict_risk_from_weather
        risk_prediction = await predict_risk_from_weather(weather_data)
        
        # Analizar factores operacionales
        operational_status = _analyze_operational_impact(
            weather_data, 
            risk_prediction
        )
        
        return {
            "airport": {
                "icao": icao.upper(),
                "name": weather_data.get("airport_name", "Unknown Airport"),
                "elevation": weather_data.get("elevation", "N/A"),
                "location": weather_data.get("location", {}),
            },
            "current_weather": {
                "timestamp": datetime.utcnow().isoformat(),
                "temperatura": weather_data.get("temperatura"),
                "humedad": weather_data.get("humedad"),
                "viento": weather_data.get("viento"),
                "visibilidad": weather_data.get("visibilidad"),
                "condicion": weather_data.get("condicion"),
                "presion": weather_data.get("presion"),
            },
            "risk_analysis": {
                "overall_risk": risk_prediction["risk_level"],
                "confidence": risk_prediction["confidence"],
                "probabilities": risk_prediction.get("probabilities", {}),
                "factors": risk_prediction.get("risk_factors", []),
            },
            "operational_impact": operational_status,
            "recommendations": risk_prediction.get("recommendations", []),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estado de aeropuerto {icao}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener estado del aeropuerto: {str(e)}"
        )


@router.get("/airport/{icao}/forecast")
async def get_airport_forecast_dashboard(
    icao: str = Depends(validate_icao_code),
    hours: int = Query(default=6, ge=1, le=24, description="Horas de pronóstico")
):
    """
    Pronóstico y tendencia de riesgo para un aeropuerto
    
    Args:
        icao: Código ICAO del aeropuerto
        hours: Horas de pronóstico (1-24)
        
    Returns:
        Pronóstico meteorológico y tendencia de riesgo
    """
    try:
        # TODO: Implementar pronóstico real con TAF
        
        return {
            "airport": icao.upper(),
            "forecast_period": f"{hours} hours",
            "risk_trend": "ESTABLE",  # MEJORANDO, ESTABLE, EMPEORANDO
            "forecast": {
                "next_6h": {
                    "expected_risk": "BAJO",
                    "weather_summary": "Condiciones favorables",
                    "alerts": []
                },
                "next_12h": {
                    "expected_risk": "BAJO",
                    "weather_summary": "Sin cambios significativos",
                    "alerts": []
                },
                "next_24h": {
                    "expected_risk": "MODERADO",
                    "weather_summary": "Posible incremento de viento",
                    "alerts": ["Viento en aumento"]
                }
            },
            "message": "Pronóstico basado en datos históricos - TAF en desarrollo"
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo pronóstico de {icao}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener pronóstico: {str(e)}"
        )


@router.get("/alerts")
async def get_active_alerts(
    severity: Optional[str] = Query(None, regex="^(LOW|MODERATE|HIGH|CRITICAL)$"),
    db: Session = Depends(get_db)
):
    """
    Obtiene alertas activas del sistema
    
    Args:
        severity: Filtrar por severidad (LOW, MODERATE, HIGH, CRITICAL)
        db: Sesión de base de datos
        
    Returns:
        Lista de alertas activas
    """
    try:
        # TODO: Implementar sistema de alertas en BD
        
        return {
            "total_alerts": 0,
            "alerts": [],
            "severity_filter": severity,
            "message": "Sistema de alertas en desarrollo"
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo alertas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener alertas: {str(e)}"
        )


@router.get("/statistics")
async def get_system_statistics(
    period_days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """
    Estadísticas generales del sistema
    
    Args:
        period_days: Días a analizar (1-30)
        db: Sesión de base de datos
        
    Returns:
        Estadísticas agregadas del sistema
    """
    try:
        # TODO: Implementar análisis real de BD
        
        return {
            "period": {
                "days": period_days,
                "start_date": (datetime.utcnow() - timedelta(days=period_days)).isoformat(),
                "end_date": datetime.utcnow().isoformat()
            },
            "predictions": {
                "total": 0,
                "by_risk_level": {
                    "BAJO": 0,
                    "MODERADO": 0,
                    "ALTO": 0,
                    "CRÍTICO": 0
                }
            },
            "airports": {
                "monitored": 0,
                "active_alerts": 0
            },
            "model_performance": {
                "average_confidence": 0.0,
                "accuracy": 0.0
            },
            "message": "Estadísticas basadas en datos históricos simulados"
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener estadísticas: {str(e)}"
        )


@router.get("/recent-predictions")
async def get_recent_predictions(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Obtiene las predicciones más recientes del sistema
    
    Args:
        limit: Número de predicciones a retornar
        db: Sesión de base de datos
        
    Returns:
        Lista de predicciones recientes
    """
    try:
        # TODO: Consultar BD real
        
        return {
            "total": 0,
            "limit": limit,
            "predictions": [],
            "message": "Historial de predicciones en desarrollo"
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo predicciones recientes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener predicciones: {str(e)}"
        )


@router.get("/demo")
async def get_demo_dashboard():
    """
    Dashboard de demostración con datos de ejemplo
    
    Útil para presentaciones y testing
    """
    return {
        "system_status": {
            "status": "operational",
            "version": "1.0.0",
            "uptime": "24h 15m"
        },
        "featured_airport": {
            "icao": "SKBO",
            "name": "El Dorado International Airport - Bogotá",
            "current_risk": "BAJO",
            "confidence": 0.87,
            "weather": {
                "temperatura": 18.5,
                "viento": 8,
                "visibilidad": 9000,
                "condicion": "Parcialmente nublado"
            }
        },
        "recent_alerts": [
            {
                "airport": "KJFK",
                "severity": "MODERATE",
                "message": "Vientos cruzados reportados",
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat()
            }
        ],
        "statistics_7d": {
            "total_predictions": 342,
            "average_confidence": 0.84,
            "risk_distribution": {
                "BAJO": 245,
                "MODERADO": 78,
                "ALTO": 17,
                "CRÍTICO": 2
            }
        },
        "note": "Dashboard de demostración con datos simulados"
    }


# ==================== HELPER FUNCTIONS ====================

def _analyze_operational_impact(weather_data: dict, risk_prediction: dict) -> dict:
    """
    Analiza el impacto operacional basado en clima y riesgo
    
    Args:
        weather_data: Datos meteorológicos actuales
        risk_prediction: Predicción de riesgo del modelo ML
        
    Returns:
        Análisis de impacto operacional
    """
    risk_level = risk_prediction.get("risk_level", "BAJO")
    viento = weather_data.get("viento", 0)
    visibilidad = weather_data.get("visibilidad", 10000)
    
    # Determinar estado de operaciones
    if risk_level == "CRÍTICO":
        departures_status = "RESTRINGIDO"
        arrivals_status = "RESTRINGIDO"
    elif risk_level == "ALTO":
        departures_status = "PRECAUCIÓN"
        arrivals_status = "PRECAUCIÓN"
    else:
        departures_status = "NORMAL"
        arrivals_status = "NORMAL"
    
    # Identificar restricciones
    restrictions = []
    if viento > 20:
        restrictions.append("Vientos fuertes - precaución en despegues/aterrizajes")
    if visibilidad < 5000:
        restrictions.append("Visibilidad reducida - procedimientos IFR")
    
    return {
        "departures": departures_status,
        "arrivals": arrivals_status,
        "restrictions": restrictions,
        "affected_operations": len(restrictions) > 0
    }