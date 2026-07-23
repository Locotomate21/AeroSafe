"""
Capa de persistencia: WeatherRepository.

Se prueba contra SQLite en memoria, sin mocks del ORM: un mock de
SQLAlchemy validaría que llamamos a los métodos, no que las consultas
devuelven lo correcto.
"""
from datetime import datetime, timedelta, timezone

import pytest

from database.repositories.weather_repository import WeatherRepository
from models.models import RiskPrediction, WeatherRecord


@pytest.fixture
def repo(db_session):
    return WeatherRepository(db_session)


def _clima(**overrides):
    datos = {
        "ciudad": "Bogotá",
        "icao": "SKBO",
        "temperatura": 18.0,
        "humedad": 70.0,
        "viento": 10.0,
        "visibilidad": 9000.0,
    }
    datos.update(overrides)
    return datos


# =========================================================================
# Registros meteorológicos
# =========================================================================

def test_crear_y_recuperar_por_id(repo):
    registro = repo.create_weather_record(_clima())

    assert registro.id is not None
    assert repo.get_weather_by_id(registro.id).ciudad == "Bogotá"


def test_get_weather_by_id_inexistente(repo):
    assert repo.get_weather_by_id(9999) is None


def test_latest_weather_ordena_por_recencia(repo, db_session):
    for i, temp in enumerate([10.0, 20.0, 30.0]):
        registro = repo.create_weather_record(_clima(temperatura=temp))
        registro.timestamp = datetime.now(timezone.utc) - timedelta(hours=3 - i)
    db_session.commit()

    ultimos = repo.get_latest_weather("Bogotá", limit=2)

    assert len(ultimos) == 2
    assert ultimos[0].temperatura == 30.0


def test_latest_weather_filtra_por_ciudad(repo):
    repo.create_weather_record(_clima(ciudad="Bogotá"))
    repo.create_weather_record(_clima(ciudad="Medellín"))

    assert len(repo.get_latest_weather("Bogotá")) == 1


def test_condiciones_peligrosas_detecta_baja_visibilidad(repo, db_session):
    repo.create_weather_record(_clima(visibilidad=9000.0, viento=5.0))
    repo.create_weather_record(_clima(visibilidad=800.0, viento=5.0))
    db_session.commit()

    peligrosas = repo.get_dangerous_conditions(hours=24)

    assert len(peligrosas) == 1
    assert peligrosas[0].visibilidad == 800.0


def test_delete_old_records(repo, db_session):
    viejo = repo.create_weather_record(_clima())
    viejo.timestamp = datetime.now(timezone.utc) - timedelta(days=60)
    repo.create_weather_record(_clima())
    db_session.commit()

    borrados = repo.delete_old_records(days=30)

    assert borrados == 1
    assert db_session.query(WeatherRecord).count() == 1


# =========================================================================
# Predicciones de riesgo
# =========================================================================

def test_crear_prediccion(repo):
    prediccion = repo.create_risk_prediction({
        "ciudad": "Bogotá",
        "icao": "SKBO",
        "riesgo": "ALTO",
        "confianza": 0.92,
        "probabilidades": {"BAJO": 0.02, "MODERADO": 0.06, "ALTO": 0.92},
    })

    assert prediccion.id is not None
    assert prediccion.riesgo == "ALTO"
    # El JSON debe sobrevivir el viaje de ida y vuelta a la base.
    assert prediccion.probabilidades["ALTO"] == 0.92


def test_predicciones_por_aeropuerto(repo):
    repo.create_risk_prediction({"icao": "SKBO", "riesgo": "BAJO", "confianza": 0.9})
    repo.create_risk_prediction({"icao": "SKRG", "riesgo": "ALTO", "confianza": 0.8})

    assert len(repo.get_predictions_by_airport("SKBO")) == 1


def test_estadisticas_de_riesgo(repo, db_session):
    """
    Nota: este método devuelve 'total_predicciones' y 'distribucion'
    (español, anidado), mientras que el endpoint GET /api/v1/risk/stats
    responde 'total_predictions' y 'risk_distribution' (inglés, plano).
    Son dos contratos distintos para lo mismo; conviene unificarlos.
    """
    for riesgo, confianza in [("BAJO", 0.9), ("BAJO", 0.8), ("ALTO", 0.7)]:
        repo.create_risk_prediction({
            "icao": "SKBO", "riesgo": riesgo, "confianza": confianza,
        })
    db_session.commit()

    stats = repo.get_risk_statistics(icao="SKBO", days=7)

    assert stats["total_predicciones"] == 3
    assert stats["distribucion"]["BAJO"]["cantidad"] == 2
    assert stats["distribucion"]["BAJO"]["porcentaje"] == pytest.approx(66.67)
    assert stats["distribucion"]["ALTO"]["cantidad"] == 1


# =========================================================================
# Aeropuertos
# =========================================================================

def test_crear_y_buscar_aeropuerto(repo):
    repo.create_airport({
        "icao": "SKBO",
        "iata": "BOG",
        "nombre": "El Dorado",
        "ciudad": "Bogotá",
        "pais": "CO",
        "latitud": 4.7016,
        "longitud": -74.1469,
    })

    encontrado = repo.get_airport("SKBO")

    assert encontrado is not None
    assert encontrado.nombre == "El Dorado"


def test_get_airport_inexistente(repo):
    assert repo.get_airport("XXXX") is None


def test_listar_aeropuertos(repo):
    repo.create_airport({"icao": "SKBO", "nombre": "El Dorado", "ciudad": "Bogotá"})
    repo.create_airport({"icao": "SKRG", "nombre": "JMC", "ciudad": "Rionegro"})

    assert len(repo.get_all_airports()) == 2
