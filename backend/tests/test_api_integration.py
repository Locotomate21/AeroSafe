"""
Tests de integración sobre la API HTTP real.

Los tests de unidad no habrían detectado el fallo principal del proyecto:
la API respondía 200 con un riesgo plausible mientras la predicción venía
de reglas if/else. Estos tests recorren el camino completo —request HTTP,
routing, serialización, pipeline de features, modelo— y comprueban en la
propia respuesta que la predicción proviene del modelo entrenado.
"""
import pytest
from fastapi.testclient import TestClient

from main import app
from services.ml_service_v2 import ml_service_v2


# TestClient sin context manager: no dispara los eventos de arranque, así
# los tests no crean el fichero SQLite ni tocan la base de datos real.
client = TestClient(app)


requiere_modelo = pytest.mark.skipif(
    ml_service_v2 is None or not ml_service_v2.can_infer(),
    reason="Modelo de producción no disponible",
)


TORMENTA = {
    "temperatura": 15.5, "humedad": 95, "viento": 45,
    "visibilidad": 800, "presion": 990, "condicion": "tormenta",
}
DESPEJADO = {
    "temperatura": 18.5, "humedad": 60, "viento": 8,
    "visibilidad": 9999, "presion": 1026, "condicion": "despejado",
}


# =========================================================================
# Endpoints básicos
# =========================================================================

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "AeroSafe"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_openapi_se_genera():
    """Si algún schema está mal definido, esto falla."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/api/v1/risk/predict" in r.json()["paths"]


# =========================================================================
# Predicción
# =========================================================================

@requiere_modelo
def test_predict_usa_el_modelo_entrenado():
    """
    El test que faltaba.

    Un 200 con un riesgo razonable no prueba nada: el modo mock también
    los devolvía. Lo que se comprueba es el origen de la predicción.
    """
    r = client.post("/api/v1/risk/predict", json=TORMENTA)

    assert r.status_code == 200
    body = r.json()

    assert body["model_status"] == "ml", (
        f"La API respondió con predicción mock: {body.get('warning')}"
    )
    assert body.get("warning") is None


@requiere_modelo
def test_predict_probabilidades_reales():
    """Las probabilidades vienen de predict_proba y suman 1."""
    body = client.post("/api/v1/risk/predict", json=TORMENTA).json()

    probs = body["probabilidades"]
    assert set(probs) == {"BAJO", "MODERADO", "ALTO"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert body["confianza"] == pytest.approx(max(probs.values()))


@requiere_modelo
def test_predict_no_expone_clase_critico():
    """
    El modelo tiene 3 clases. Prometer CRÍTICO en la respuesta era ofrecer
    algo que nunca podía ocurrir.
    """
    body = client.post("/api/v1/risk/predict", json=TORMENTA).json()

    assert "CRÍTICO" not in body["probabilidades"]
    assert body["riesgo"] in {"BAJO", "MODERADO", "ALTO"}


@requiere_modelo
def test_predict_declara_features_imputadas():
    """La respuesta dice qué variables tuvo que estimar."""
    body = client.post("/api/v1/risk/predict", json=DESPEJADO).json()

    assert isinstance(body["imputed_features"], list)
    assert "turbulencia" in body["imputed_features"]


@requiere_modelo
def test_predict_tormenta_es_alto():
    body = client.post("/api/v1/risk/predict", json=TORMENTA).json()
    assert body["riesgo"] == "ALTO"


@requiere_modelo
def test_predict_despejado_es_bajo():
    body = client.post("/api/v1/risk/predict", json=DESPEJADO).json()
    assert body["riesgo"] == "BAJO"


@requiere_modelo
def test_demo_devuelve_prediccion_real():
    """
    Regresión: /demo devolvía una predicción inventada con confianza 0.85
    si algo fallaba, indistinguible de una real.
    """
    r = client.get("/api/v1/risk/demo")

    assert r.status_code == 200
    assert r.json()["model_status"] == "ml"


# =========================================================================
# Validación de entrada
# =========================================================================

@pytest.mark.parametrize("payload,campo", [
    ({"humedad": 60, "viento": 8, "visibilidad": 9999}, "temperatura"),
    ({"temperatura": 20, "viento": 8, "visibilidad": 9999}, "humedad"),
    ({"temperatura": 20, "humedad": 60, "visibilidad": 9999}, "viento"),
])
def test_predict_rechaza_payload_incompleto(payload, campo):
    r = client.post("/api/v1/risk/predict", json=payload)
    assert r.status_code == 422
    assert campo in r.text


@pytest.mark.parametrize("humedad", [-1, 101])
def test_predict_rechaza_humedad_fuera_de_rango(humedad):
    r = client.post("/api/v1/risk/predict", json={**DESPEJADO, "humedad": humedad})
    assert r.status_code == 422


def test_predict_rechaza_viento_negativo():
    r = client.post("/api/v1/risk/predict", json={**DESPEJADO, "viento": -5})
    assert r.status_code == 422


@pytest.mark.parametrize("icao", ["SK", "SKBOX", "SK1O"])
def test_predict_airport_rechaza_icao_invalido(icao):
    r = client.post(f"/api/v1/risk/predict/airport/{icao}")
    assert r.status_code == 400


# =========================================================================
# Endpoints informativos
# =========================================================================

def test_risk_test_endpoint():
    r = client.get("/api/v1/risk/test")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_info():
    r = client.get("/info")
    assert r.status_code == 200
    assert r.json()["project"] == "AeroSafe"
