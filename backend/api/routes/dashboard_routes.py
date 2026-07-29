"""
Endpoints de agregacion para paneles.

Se conservan solo los que hacen trabajo REAL. Se eliminaron cinco
endpoints que devolvian datos mock o fabricados (/demo inventaba "342
predicciones" y una alerta falsa; /statistics, /alerts,
/recent-predictions y /airport/{icao}/forecast devolvian ceros o texto
"en desarrollo"). Ademas duplicaban endpoints que ya existen de verdad:
/health, /api/v1/forecast/{icao}, /api/v1/risk/stats y /risk/history.

Servir datos falsos con apariencia real es justo lo que este proyecto
evita en todo lo demas.
"""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from api.dependencies import get_db, validate_icao_code
from models.models import RiskPrediction

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/status")
async def get_dashboard_status(db: Session = Depends(get_db)):
    """
    Estado agregado del sistema, con cada campo verificado de verdad.

    A diferencia de la version anterior, no reporta "database: connected"
    a ciegas ni inventa metricas: hace SELECT 1 y cuenta las predicciones
    reales en la base.
    """
    from core.config import settings
    from services.ml_service_v2 import ml_service_v2

    modelo_ok = ml_service_v2 is not None and ml_service_v2.can_infer()

    try:
        db.execute(text("SELECT 1"))
        total = db.query(func.count(RiskPrediction.id)).scalar() or 0
        db_ok = True
    except Exception as e:
        logger.error("Dashboard status: fallo de BD: %s", e)
        db_ok = False
        total = None

    return {
        "status": "operational" if (modelo_ok and db_ok) else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.VERSION,
        "components": {
            "ml_model": "operativo" if modelo_ok else "no_disponible",
            "weather_api": "configurada" if settings.OPENWEATHER_API_KEY else "sin_clave",
            "database": "connected" if db_ok else "error",
        },
        "metrics": {
            "total_predictions": total,
        },
    }


@router.get("/airport/{icao}/status")
async def get_airport_status(icao: str = Depends(validate_icao_code)):
    """
    Estado completo de un aeropuerto: clima actual, riesgo e impacto
    operacional. Agrega el servicio de clima y el clasificador de riesgo.
    """
    try:
        from services.aviation_weather_service import get_airport_weather_data
        from services.ml_service_v2 import predict_risk_from_weather

        weather_data = await get_airport_weather_data(icao)
        risk = await predict_risk_from_weather(weather_data)

        return {
            "airport": {
                "icao": icao.upper(),
                "name": weather_data.get("airport_name", "Unknown Airport"),
                "elevation": weather_data.get("elevation", "N/A"),
                "location": weather_data.get("location", {}),
            },
            "current_weather": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "temperatura": weather_data.get("temperatura"),
                "humedad": weather_data.get("humedad"),
                "viento": weather_data.get("viento"),
                "visibilidad": weather_data.get("visibilidad"),
                "condicion": weather_data.get("condicion"),
                "presion": weather_data.get("presion"),
            },
            "risk_analysis": {
                "overall_risk": risk["risk_level"],
                "confidence": risk["confidence"],
                "probabilities": risk.get("probabilities", {}),
                "factors": risk.get("risk_factors", []),
                "model_status": risk.get("model_status"),
            },
            "operational_impact": _analyze_operational_impact(weather_data, risk),
            "recommendations": risk.get("recommendations", []),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error("Error obteniendo estado de aeropuerto %s: %s", icao, e)
        raise HTTPException(status_code=502, detail="Error al obtener el estado del aeropuerto")


def _analyze_operational_impact(weather_data: dict, risk_prediction: dict) -> dict:
    """Impacto operacional a partir del clima y el nivel de riesgo."""
    risk_level = risk_prediction.get("risk_level", "BAJO")
    viento = weather_data.get("viento", 0)
    visibilidad = weather_data.get("visibilidad", 10000)

    if risk_level == "ALTO":
        estado = "PRECAUCIÓN"
    else:
        estado = "NORMAL"

    restricciones = []
    if viento > 20:
        restricciones.append("Vientos fuertes - precaución en despegues/aterrizajes")
    if visibilidad < 5000:
        restricciones.append("Visibilidad reducida - procedimientos IFR")

    return {
        "departures": estado,
        "arrivals": estado,
        "restrictions": restricciones,
        "affected_operations": len(restricciones) > 0,
    }
