"""
Endpoint de pronóstico.

Se mockea la descarga del METAR (que va a NOAA en vivo) para que los
tests no dependan de la red ni del tiempo que haga hoy en Bogotá. Todo lo
demás —parseo, features, modelo calibrado— se ejercita de verdad.
"""
import pytest
from fastapi.testclient import TestClient

from main import app
from services.forecast_service import get_forecast_service

client = TestClient(app)

# METAR reales de SKBO para distintas condiciones.
METAR_NIEBLA_MADRUGADA = "METAR SKBO 230600Z 00000KT 0500 FG OVC002 11/11 Q1027"
METAR_NIEBLA_MANANA = "METAR SKBO 231100Z 00000KT 0500 FG OVC002 11/11 Q1027"
METAR_DESPEJADO = "METAR SKBO 231500Z 09008KT 9999 SCT025 19/09 Q1026 NOSIG"

_skbo_disponible = get_forecast_service("SKBO", 3).disponible()
requiere_modelo = pytest.mark.skipif(
    not _skbo_disponible, reason="Modelo de pronóstico SKBO no disponible"
)


@pytest.fixture
def mock_metar(monkeypatch):
    """Inyecta un METAR fijo en el servicio, saltándose la red."""
    def _set(raw):
        async def fake_get(self, icao):
            return {"raw": raw}
        monkeypatch.setattr(
            "services.metar_taf_service.METARTAFService.get_metar_data", fake_get
        )
    return _set


# =========================================================================
# Contrato del endpoint
# =========================================================================

@requiere_modelo
def test_pronostico_estructura(mock_metar):
    mock_metar(METAR_NIEBLA_MADRUGADA)
    r = client.get("/api/v1/forecast/SKBO")

    assert r.status_code == 200
    body = r.json()
    assert body["icao"] == "SKBO"
    assert body["horizonte_horas"] == 3
    assert 0.0 <= body["probabilidad"] <= 1.0
    assert body["nivel"] in {"MINIMO", "BAJO", "MODERADO", "ALTO"}
    assert body["modelo_calibrado"] is True


@requiere_modelo
def test_probabilidad_calibrada_es_coherente_con_la_fisica(mock_metar):
    """
    Niebla de madrugada (1am local) persiste mucho más que la de media
    mañana (6am local, se disipa con el sol). El pronóstico debe reflejarlo.
    """
    mock_metar(METAR_NIEBLA_MADRUGADA)
    madrugada = client.get("/api/v1/forecast/SKBO").json()["probabilidad"]

    mock_metar(METAR_NIEBLA_MANANA)
    manana = client.get("/api/v1/forecast/SKBO").json()["probabilidad"]

    assert madrugada > manana, (
        f"la niebla de madrugada ({madrugada}) deberia predecir mas "
        f"persistencia que la de la manana ({manana})"
    )


@requiere_modelo
def test_despejado_da_probabilidad_minima(mock_metar):
    mock_metar(METAR_DESPEJADO)
    body = client.get("/api/v1/forecast/SKBO").json()

    assert body["probabilidad"] < 0.1
    assert body["nivel"] == "MINIMO"
    assert body["es_adverso_ahora"] is False


@requiere_modelo
def test_reporta_condicion_actual(mock_metar):
    mock_metar(METAR_NIEBLA_MADRUGADA)
    body = client.get("/api/v1/forecast/SKBO").json()

    assert body["condicion_actual"] == "niebla"
    assert body["es_adverso_ahora"] is True


@requiere_modelo
def test_incluye_el_metar_usado(mock_metar):
    mock_metar(METAR_NIEBLA_MADRUGADA)
    body = client.get("/api/v1/forecast/SKBO").json()
    assert body["metar"] == METAR_NIEBLA_MADRUGADA


# =========================================================================
# Validación y errores
# =========================================================================

def test_icao_no_soportado_da_404():
    r = client.get("/api/v1/forecast/KJFK")
    assert r.status_code == 404
    assert "Soportados" in r.json()["detail"]


def test_icao_invalido_da_error():
    # Longitud incorrecta la rechaza la validación de la ruta (422).
    assert client.get("/api/v1/forecast/SK").status_code == 422
    # 4 caracteres pero con dígitos: lo rechaza el handler (400).
    assert client.get("/api/v1/forecast/SK1O").status_code == 400


@requiere_modelo
def test_metar_incompleto_da_422(mock_metar):
    """Un METAR sin temperatura ni visibilidad no permite pronosticar."""
    mock_metar("METAR SKBO 231100Z 00000KT")
    r = client.get("/api/v1/forecast/SKBO")
    assert r.status_code == 422


def test_listado_de_soportados():
    r = client.get("/api/v1/forecast/")
    assert r.status_code == 200
    assert set(r.json()["aeropuertos_soportados"]) == {"SKBO", "SKRG", "SKPS", "SKMZ"}


@requiere_modelo
def test_openapi_incluye_forecast():
    spec = client.get("/openapi.json").json()
    assert "/api/v1/forecast/{icao}" in spec["paths"]
