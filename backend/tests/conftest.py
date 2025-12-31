import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.services.ml_service_v2 import MLServiceV2
# ← IMPORTANTE: Importar el modelo para que se registre en Base.metadata
from backend.models.models import RiskPrediction


@pytest.fixture(scope="session")
def ml_service():
    """
    Crea un servicio ML con un modelo mockeado para tests.
    """
    # Crear un mock del modelo
    mock_model = Mock()
    
    # Configurar el comportamiento del mock
    mock_model.predict.return_value = np.array(["BAJO"])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])
    mock_model.classes_ = np.array(["BAJO", "MEDIO", "ALTO"])
    
    # Retornar una instancia del servicio con el modelo mockeado
    return MLServiceV2(model=mock_model)


@pytest.fixture
def sample_raw_weather():
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
def db_session():
    """
    Crea una base de datos en memoria para tests.
    """
    engine = create_engine("sqlite:///:memory:")
    TestingSession = sessionmaker(bind=engine)

    # Crear todas las tablas (ahora incluye risk_predictions)
    Base.metadata.create_all(engine)
    session = TestingSession()

    try:
        yield session
    finally:
        session.close()


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))