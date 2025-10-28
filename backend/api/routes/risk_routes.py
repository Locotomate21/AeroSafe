# backend/routes/risk_routes.py
from fastapi import APIRouter
from backend.models.weather_model import WeatherData
from backend.services.risk_predictor import evaluate_risk
from backend.models.database import SessionLocal, Weather

router = APIRouter()

@router.post("/predict-risk")
def predict_risk(weather: WeatherData):
    """
    Recibe datos meteorológicos, evalúa el riesgo y guarda la información.
    """
    # Convertir WeatherData a formato de evaluate_risk
    risk_input = {
        "temperatura": weather.temperature,
        "viento": 10,            # valor por defecto o dinámico si se agrega viento real
        "visibilidad": 8000,     # idem
        "humedad": weather.humidity,
        "descripcion": weather.description
    }

    # Evaluar riesgo
    resultado = evaluate_risk(risk_input)

    # Guardar en la base de datos
    db = SessionLocal()
    weather_record = Weather(
        city=weather.city,
        temperature=weather.temperature,
        humidity=weather.humidity,
        description=weather.description,
        timestamp=weather.timestamp
    )
    db.add(weather_record)
    db.commit()
    db.close()

    return {"evaluacion": resultado}

# 🆕 Nuevo endpoint: obtener historial
@router.get("/history")
def get_history(limit: int = 10):
    """
    Devuelve las últimas evaluaciones meteorológicas guardadas.
    """
    db = SessionLocal()
    records = db.query(Weather).order_by(Weather.timestamp.desc()).limit(limit).all()
    db.close()

    historial = [
        {
            "city": r.city,
            "temperature": r.temperature,
            "humidity": r.humidity,
            "description": r.description,
            "timestamp": r.timestamp
        }
        for r in records
    ]
    return {"historial": historial}
