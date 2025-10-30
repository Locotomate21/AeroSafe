import pytest
from unittest.mock import patch, AsyncMock


def test_weather_test_endpoint(client):
    """Test del endpoint de prueba"""
    response = client.get("/api/v1/weather/test")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "ok"


@patch('services.weather_api_service.weather_api_service.get_current_weather')
def test_get_current_weather(mock_get_weather, client, mock_weather_api_response):
    """Test de obtener clima actual"""
    # Configurar mock
    mock_get_weather.return_value = mock_weather_api_response
    
    response = client.get("/api/v1/weather/current/Bogotá,CO")
    
    assert response.status_code == 200
    # Verificar estructura de respuesta


@patch('services.weather_api_service.weather_api_service.get_airport_weather')
def test_get_airport_weather(mock_get_airport, client):
    """Test de clima de aeropuerto"""
    mock_get_airport.return_value = {
        "airport_name": "El Dorado",
        "icao_code": "SKBO",
        "temperatura": 20.0,
        "humedad": 65,
        "viento": 15.0,
        "visibilidad": 8000
    }
    
    response = client.get("/api/v1/weather/airport/SKBO")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["icao_code"] == "SKBO"
    assert "temperatura" in data


def test_get_airport_invalid_icao(client):
    """Test con código ICAO inválido"""
    response = client.get("/api/v1/weather/airport/INVALID")
    
    assert response.status_code == 400  # Bad request