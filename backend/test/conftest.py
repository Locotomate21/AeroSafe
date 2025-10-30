import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from database.connection import get_db
from models.database import Base

# ==================== DATABASE TEST ====================

# BD en memoria para tests (rápida y limpia)
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """
    Crea una sesión de BD limpia para cada test
    
    Uso:
        def test_something(db_session):
            user = User(name="Test")
            db_session.add(user)
            db_session.commit()
    """
    # Crear tablas
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Limpiar tablas después del test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """
    Cliente de FastAPI con BD de test
    
    Uso:
        def test_endpoint(client):
            response = client.get("/health")
            assert response.status_code == 200
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


# ==================== DATA FIXTURES ====================

@pytest.fixture
def sample_weather_data():
    """Datos de ejemplo para tests"""
    return {
        "temperatura": 20.0,
        "humedad": 65.0,
        "viento": 15.0,
        "visibilidad": 8000.0
    }


@pytest.fixture
def sample_risk_request():
    """Request de riesgo de ejemplo"""
    return {
        "temperatura": 15.5,
        "humedad": 85,
        "viento": 35,
        "visibilidad": 2500
    }


@pytest.fixture
def dangerous_weather_data():
    """Datos de clima peligroso"""
    return {
        "temperatura": 5.0,
        "humedad": 95.0,
        "viento": 45.0,
        "visibilidad": 1500.0
    }


@pytest.fixture
def safe_weather_data():
    """Datos de clima seguro"""
    return {
        "temperatura": 22.0,
        "humedad": 60.0,
        "viento": 10.0,
        "visibilidad": 9000.0
    }


@pytest.fixture
def sample_airport():
    """Aeropuerto de ejemplo"""
    return {
        "icao": "SKBO",
        "iata": "BOG",
        "nombre": "El Dorado",
        "ciudad": "Bogotá",
        "pais": "Colombia",
        "latitud": 4.7016,
        "longitud": -74.1469,
        "elevacion": 2548.0,
        "activo": True
    }


# ==================== MOCK FIXTURES ====================

@pytest.fixture
def mock_weather_api_response():
    """Respuesta simulada de API de clima"""
    return {
        "main": {
            "temp": 20.0,
            "feels_like": 19.0,
            "temp_min": 18.0,
            "temp_max": 22.0,
            "pressure": 1013,
            "humidity": 65,
            "sea_level": 1013
        },
        "wind": {
            "speed": 4.2,  # m/s
            "deg": 180,
            "gust": 6.5
        },
        "visibility": 10000,
        "clouds": {"all": 20},
        "weather": [
            {
                "main": "Clear",
                "description": "clear sky"
            }
        ],
        "coord": {"lat": 4.7016, "lon": -74.1469},
        "name": "Bogotá",
        "sys": {"country": "CO"},
        "dt": 1698000000
    }