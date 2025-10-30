import pytest
from services.ml_service import ml_service


def test_ml_service_loads():
    """Test que el servicio ML carga correctamente"""
    assert ml_service is not None
    assert hasattr(ml_service, 'model')
    assert hasattr(ml_service, 'scaler')
    assert hasattr(ml_service, 'encoder')


def test_predict_risk_safe(safe_weather_data):
    """Test de predicción con datos seguros"""
    result = ml_service.predict_risk(safe_weather_data)
    
    assert result["risk_level"] == "Seguro"
    assert result["confidence"] > 0.5
    assert "Seguro" in result["probabilities"]
    assert "Precaución" in result["probabilities"]
    assert "Peligro" in result["probabilities"]


def test_predict_risk_dangerous(dangerous_weather_data):
    """Test de predicción con datos peligrosos"""
    result = ml_service.predict_risk(dangerous_weather_data)
    
    assert result["risk_level"] == "Peligro"
    assert result["confidence"] > 0.7


def test_predict_risk_precaution():
    """Test de predicción con precaución"""
    data = {
        "temperatura": 18.0,
        "humedad": 85.0,
        "viento": 28.0,
        "visibilidad": 4500.0
    }
    
    result = ml_service.predict_risk(data)
    
    assert result["risk_level"] in ["Precaución", "Peligro"]
    assert 0 <= result["confidence"] <= 1


def test_probabilities_sum_to_one(sample_weather_data):
    """Test que las probabilidades suman 1.0"""
    result = ml_service.predict_risk(sample_weather_data)
    
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.01  # Tolerancia de error