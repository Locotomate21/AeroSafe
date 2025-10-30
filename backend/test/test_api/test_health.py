import pytest

def test_root_endpoint(client):
    """Test del endpoint raíz"""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_health_check(client):
    """Test del health check"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "service" in data


def test_docs_available(client):
    """Test que /docs esté disponible"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_available(client):
    """Test que /openapi.json esté disponible"""
    response = client.get("/openapi.json")
    assert response.status_code == 200