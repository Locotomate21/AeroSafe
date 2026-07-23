import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock

# Agregar backend/ al path ANTES de importar módulos de la aplicación.
#   conftest.py -> tests/ -> backend/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from database import Base
from features.build_features import build_features
from services.ml_service_v2 import MLServiceV2
from models.models import RiskPrediction


# =========================================================================
# Datos base
# =========================================================================
# Nombres canónicos del schema (features/schema.py). Los tests deben usar
# estos: si un fixture inventa nombres como 'rafaga' o 'nubes', prueba un
# contrato que el modelo no tiene.

def _fila(**overrides):
    """Una observación completa, con las 26 features base del modelo."""
    fila = {
        "temperatura": 20.0,
        "humedad": 80.0,
        "presion": 1013.0,
        "viento": 7.0,
        "visibilidad": 6000.0,
        "precipitacion": 0.0,
        "direccion_viento": 90.0,
        "runway_heading": 134.0,
        "viento_cruzado": 5.0,
        "viento_frente": 4.9,
        "rafagas": 10.0,
        "techo_nubes": 2000.0,
        "tipo_nubes": "nublado",
        "turbulencia": "leve",
        "estado_pista": "humeda",
        "descripcion": "nublado",
        "altitud_aeropuerto": 2548.0,
        "altitud_densidad": 3000.0,
        "punto_rocio": 16.5,
        "riesgo_hielo": 0,
        "tormenta_electrica": 0,
        "cizalladura_viento": 0,
        "hora": 12,
        "es_noche": 0,
        "mes": 6,
        "dia_año": 180,
    }
    fila.update(overrides)
    return fila


@pytest.fixture
def fila_completa():
    """Un DataFrame de una fila con todas las features base."""
    return pd.DataFrame([_fila()])


@pytest.fixture
def sample_raw_weather():
    """Payload parcial, tal como llega desde la API (solo lo observable)."""
    return pd.DataFrame([{
        "temperatura": 22.0,
        "humedad": 60.0,
        "presion": 1013.0,
        "viento": 5.0,
        "visibilidad": 8000.0,
    }])


@pytest.fixture
def sample_batch_data():
    """Tres observaciones para batch, con nombres canónicos."""
    return pd.DataFrame([
        _fila(ciudad="Bogotá", temperatura=20.0, visibilidad=6000.0),
        _fila(ciudad="Bogotá", temperatura=14.0, visibilidad=900.0,
              descripcion="niebla", viento=3.0),
        _fila(ciudad="Bogotá", temperatura=16.0, visibilidad=2500.0,
              descripcion="tormenta", viento=45.0, tormenta_electrica=1),
    ])


# =========================================================================
# Servicios ML
# =========================================================================

@pytest.fixture(scope="session")
def pipeline_ajustado():
    """
    Ajusta scaler y encoders sobre una muestra sintética.

    Un modelo mockeado no basta: build_features exige el scaler y los
    encoders del entrenamiento, precisamente para que nadie infiera con
    transformaciones sin ajustar. El fixture replica ese contrato.
    """
    muestra = pd.DataFrame([
        _fila(descripcion=d, tipo_nubes=n, turbulencia=t, estado_pista=p,
              temperatura=temp, viento=v, visibilidad=vis)
        for d, n, t, p, temp, v, vis in [
            ("despejado", "despejado", "ninguna", "seca", 25.0, 5.0, 9999.0),
            ("nublado", "dispersas", "leve", "humeda", 18.0, 12.0, 7000.0),
            ("lluvia_ligera", "nublado", "leve", "humeda", 15.0, 18.0, 4000.0),
            ("lluvia_fuerte", "cubierto", "moderada", "mojada", 12.0, 30.0, 2000.0),
            ("tormenta", "cubierto", "severa", "contaminada", 10.0, 45.0, 800.0),
            ("niebla", "cubierto", "ninguna", "humeda", 8.0, 2.0, 500.0),
            ("nieve", "cubierto", "leve", "contaminada", -2.0, 20.0, 1500.0),
            ("granizo", "cubierto", "severa", "contaminada", 14.0, 35.0, 3000.0),
        ]
    ])
    _, artifacts = build_features(muestra, fit=True)
    return artifacts


@pytest.fixture(scope="session")
def ml_service(pipeline_ajustado):
    """
    Servicio ML con modelo mockeado pero pipeline de features REAL.

    Así los tests ejercitan complete_raw_features + build_features de
    verdad, y solo se simula la parte cara (el estimador).
    """
    mock_model = MagicMock()
    clases = np.array(["ALTO", "BAJO", "MODERADO"])  # orden alfabético, como sklearn

    mock_model.predict = lambda X: np.array(["BAJO"] * len(X))
    mock_model.predict_proba = lambda X: np.array([[0.05, 0.80, 0.15]] * len(X))
    mock_model.classes_ = clases

    return MLServiceV2(
        model=mock_model,
        scaler=pipeline_ajustado["scaler"],
        encoders=pipeline_ajustado["encoders"],
    )


@pytest.fixture(scope="session")
def modelo_produccion():
    """
    Servicio ML con el modelo real de models/production/.

    Se omite el test si el modelo no está presente (por ejemplo en un CI
    que no descarga los artefactos), en vez de fallar.
    """
    service = MLServiceV2()
    if not service.can_infer() or service.scaler is None:
        pytest.skip("Modelo de producción no disponible en models/production/")
    return service


# =========================================================================
# Base de datos
# =========================================================================

@pytest.fixture
def db_session():
    """Base de datos SQLite en memoria, aislada por test."""
    engine = create_engine("sqlite:///:memory:")
    TestingSession = sessionmaker(bind=engine)

    Base.metadata.create_all(engine)
    session = TestingSession()

    try:
        yield session
    finally:
        session.close()
