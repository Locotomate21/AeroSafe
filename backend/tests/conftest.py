import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# CORRECCIÓN: Agregar el path ANTES de importar módulos de backend
ROOT = Path(__file__).resolve().parents[2]  # Sube 2 niveles desde backend/test/conftest.py
sys.path.insert(0, str(ROOT))  # Usar insert(0) en lugar de append para darle prioridad

from backend.database import Base
from backend.services.ml_service_v2 import MLServiceV2
from backend.models.models import RiskPrediction


@pytest.fixture(scope="session")
def ml_service():
    """
    Crea un servicio ML con un modelo mockeado para tests.
    Este mock simula correctamente el comportamiento del modelo
    sin depender del modelo real de producción.
    """
    # Crear un mock del modelo
    mock_model = MagicMock()
    
    # Configurar el comportamiento del mock
    # El mock acepta CUALQUIER número de features (para tests)
    def mock_predict(X):
        # Retornar predicciones basadas en el número de filas
        return np.array(["BAJO"] * len(X))
    
    def mock_predict_proba(X):
        # Retornar probabilidades basadas en el número de filas
        return np.array([[0.8, 0.15, 0.05]] * len(X))
    
    mock_model.predict = mock_predict
    mock_model.predict_proba = mock_predict_proba
    mock_model.classes_ = np.array(["BAJO", "MEDIO", "ALTO"])
    
    # Retornar una instancia del servicio con el modelo mockeado
    return MLServiceV2(model=mock_model)


@pytest.fixture
def sample_raw_weather():
    """Datos de prueba con formato de API externa (inglés)."""
    return pd.DataFrame([{
        "temp": 22.0,
        "humidity": 60.0,
        "pressure": 1013.0,
        "wind_speed": 5.0,
        "wind_gust": 8.0,
        "visibility": 8000.0,
        "precipitation": 0.0,
        "clouds": 40.0,
        "ice_risk": 0,
    }])


@pytest.fixture
def sample_batch_data():
    """Datos de prueba para batch processing (español)."""
    return pd.DataFrame([
        {
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
        },
        {
            "ciudad": "Medellín",
            "temperatura": 22.0,
            "humedad": 75.0,
            "presion": 1015.0,
            "viento": 5.0,
            "rafaga": 8.0,
            "visibilidad": 8000.0,
            "precipitacion": 0.0,
            "nubes": 40.0,
            "hielo": 0,
        },
        {
            "ciudad": "Cali",
            "temperatura": 25.0,
            "humedad": 70.0,
            "presion": 1010.0,
            "viento": 10.0,
            "rafaga": 15.0,
            "visibilidad": 5000.0,
            "precipitacion": 2.0,
            "nubes": 80.0,
            "hielo": 0,
        }
    ])


@pytest.fixture
def db_session():
    """
    Crea una base de datos en memoria para tests.
    """
    engine = create_engine("sqlite:///:memory:")
    TestingSession = sessionmaker(bind=engine)

    # Crear todas las tablas (incluye risk_predictions)
    Base.metadata.create_all(engine)
    session = TestingSession()

    try:
        yield session
    finally:
        session.close()