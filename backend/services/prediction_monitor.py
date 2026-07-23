from database.connection import SessionLocal
from models.models import RiskPrediction


def log_prediction(
    *,
    input_data: dict,
    prediction: dict,
    ciudad: str,
    icao: str | None = None
):
    db = SessionLocal()
    try:
        record = RiskPrediction(
            ciudad=ciudad,
            icao=icao,
            riesgo=prediction["risk_level"],
            confianza=prediction["confidence"],
            probabilidades=prediction["probabilities"],
            temperatura=input_data["temperatura"],
            humedad=input_data["humedad"],
            viento=input_data["viento"],
            visibilidad=input_data["visibilidad"],
        )
        db.add(record)
        db.commit()
    finally:
        db.close()
