from backend.models.models import RiskPrediction  # ← Asegúrate que esté así


def test_prediction_is_logged(db_session, ml_service):
    payload = {
        "ciudad": "Bogotá",
        "temperatura": 20.0,
        "humedad": 80.0,
        "presion": 1013.0,
        "viento": 7.0,
        "rafaga": 10.0,
        "visibilidad": 6000.0,
        "precipitacion": 0.0,
        "nubes": 50.0,
        "hielo": 0,
    }

    result = ml_service.predict(payload, db=db_session)

    # Usar el ORM en lugar de SQL textual
    rows = db_session.query(RiskPrediction).count()

    assert rows == 1
    
    # Validaciones adicionales
    prediction = db_session.query(RiskPrediction).first()
    assert prediction.ciudad == "Bogotá"
    assert prediction.riesgo == "BAJO"
    assert prediction.confianza == 0.8