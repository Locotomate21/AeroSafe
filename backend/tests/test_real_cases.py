"""
Validación operacional del modelo de producción.

Estos tests no miden accuracy (eso es trabajo de MLflow sobre el conjunto
de test): comprueban que el modelo servido a través de la API se comporta
de forma defendible ante condiciones reconocibles por cualquier
despachador. Un modelo que llame BAJO a una tormenta con 1200 m de
visibilidad no debe llegar a producción aunque su accuracy sea 0.99.

Se omiten si models/production/ no está disponible.
"""
import pandas as pd
import pytest

RIESGO_ORDEN = {"BAJO": 0, "MODERADO": 1, "ALTO": 2}


def _predecir(service, **wx):
    """Predice sobre un payload parcial, como hace la API."""
    df = pd.DataFrame([wx])
    return service.predict_batch(df, icao="SKBO").iloc[0].to_dict()


def test_usa_el_modelo_y_no_el_mock(modelo_produccion):
    """
    La barrera principal: si esto falla, la API está devolviendo reglas
    if/else disfrazadas de modelo entrenado.
    """
    r = _predecir(
        modelo_produccion,
        temperatura=18.5, humedad=60, viento=8,
        visibilidad=9999, presion=1026, descripcion="despejado",
    )

    assert r["model_status"] == "ml"
    assert r["confianza"] > 0.0


def test_probabilidades_son_reales(modelo_produccion):
    """
    Las probabilidades vienen de predict_proba, no de una tabla fija.

    Regresión: _get_probabilities_dict() devolvía constantes escritas a
    mano según el nivel de riesgo.
    """
    r = _predecir(
        modelo_produccion,
        temperatura=18.5, humedad=60, viento=8,
        visibilidad=9999, presion=1026, descripcion="despejado",
    )

    probs = [r["prob_BAJO"], r["prob_MODERADO"], r["prob_ALTO"]]

    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert r["confianza"] == pytest.approx(max(probs))


def test_dia_despejado_es_riesgo_bajo(modelo_produccion):
    """Condiciones VMC en SKBO: operación normal."""
    r = _predecir(
        modelo_produccion,
        temperatura=18.5, humedad=60, viento=8,
        visibilidad=9999, presion=1026, descripcion="despejado",
    )
    assert r["riesgo"] == "BAJO"


def test_tormenta_severa_es_riesgo_alto(modelo_produccion):
    """
    Caso tipo: METAR SKBO 27020G35KT 1200 +TSRA
    Viento 20kt racheado a 35kt, visibilidad 1200 m, tormenta con lluvia
    fuerte. Cualquier salida distinta de ALTO es inaceptable.
    """
    r = _predecir(
        modelo_produccion,
        temperatura=15.0, humedad=95, viento=37, rafagas=65,
        direccion_viento=270, visibilidad=1200, presion=995,
        descripcion="tormenta", tormenta_electrica=1,
    )

    assert r["riesgo"] == "ALTO"
    assert r["confianza"] > 0.6


def test_niebla_bajo_minimos_no_es_riesgo_bajo(modelo_produccion):
    """
    600 m de visibilidad está por debajo del mínimo CAT I (550 m de RVR)
    con muy poco margen. Etiquetar esto como BAJO sería un fallo de
    seguridad, no un error de precisión.
    """
    r = _predecir(
        modelo_produccion,
        temperatura=12.0, humedad=99, viento=3,
        visibilidad=600, presion=1015, descripcion="niebla",
    )

    assert r["riesgo"] != "BAJO"


def test_el_riesgo_no_baja_al_empeorar_la_visibilidad(modelo_produccion):
    """
    Monotonía: degradando solo la visibilidad, el riesgo no puede mejorar.
    Detecta modelos que aprendieron correlaciones espurias.
    """
    niveles = []
    for visibilidad in (9999, 5000, 2000, 800):
        r = _predecir(
            modelo_produccion,
            temperatura=15.0, humedad=85, viento=20,
            visibilidad=visibilidad, presion=1010, descripcion="nublado",
        )
        niveles.append(RIESGO_ORDEN[r["riesgo"]])

    assert niveles == sorted(niveles), (
        f"El riesgo baja al empeorar la visibilidad: {niveles}"
    )


def test_features_imputadas_se_reportan(modelo_produccion):
    """
    Una predicción hecha sobre valores estimados debe decirlo.
    """
    df = pd.DataFrame([{
        "temperatura": 18.0, "humedad": 70, "viento": 10,
        "visibilidad": 8000, "presion": 1015,
    }])
    resultado = modelo_produccion.predict_batch(df, icao="SKBO")

    imputadas = resultado.attrs["imputed_features"]
    assert "turbulencia" in imputadas
    assert "estado_pista" in imputadas
