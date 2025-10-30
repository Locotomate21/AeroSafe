import pytest


def test_predict_risk_safe(client, safe_weather_data):
    """Test de predicción con clima seguro"""
    response = client.post("/api/v1/risk/predict", json=safe_weather_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "riesgo" in data
    assert "confianza" in data
    assert "probabilidades" in data
    assert data["riesgo"] == "Seguro"
    assert data["confianza"] > 0.7


def test_predict_risk_dangerous(client, dangerous_weather_data):
    """Test de predicción con clima peligroso"""
    response = client.post("/api/v1/risk/predict", json=dangerous_weather_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["riesgo"] == "Peligro"
    assert data["confianza"] > 0.8


def test_predict_risk_missing_fields(client):
    """Test con campos faltantes"""
    incomplete_data = {
        "temperatura": 20.0,
        "humedad": 65.0
        # Faltan viento y visibilidad
    }
    
    response = client.post("/api/v1/risk/predict", json=incomplete_data)
    assert response.status_code == 422  # Validation error


def test_predict_risk_invalid_values(client):
    """Test con valores inválidos"""
    invalid_data = {
        "temperatura": 20.0,
        "humedad": 150.0,  # Inválido (>100)
        "viento": 15.0,
        "visibilidad": 8000.0
    }
    
    response = client.post("/api/v1/risk/predict", json=invalid_data)
    assert response.status_code == 422


def test_risk_test_endpoint(client):
    """Test del endpoint de prueba"""
    response = client.get("/api/v1/risk/test")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "ok"