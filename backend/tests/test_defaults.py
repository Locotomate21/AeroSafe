"""
Completado de features y fallback a mock.

Dos cosas que este proyecto aprendió por las malas:

1. Imputar con un perfil benigno global hacía que niebla bajo mínimos
   saliera BAJO: veinte señales de "día perfecto" ahogaban a la mala.
2. Un fallback a mock silencioso es peor que un error, porque la
   respuesta es indistinguible de una predicción real.
"""
import math

import pandas as pd
import pytest

from features.defaults import (
    AIRPORTS,
    PERFILES,
    altitud_densidad,
    complete_raw_features,
    punto_rocio,
    riesgo_hielo,
    viento_cruzado,
    viento_frente,
)
from features.build_features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from services.ml_service_v2 import MODEL_STATUS_MOCK, MLServiceV2


PAYLOAD_MINIMO = {
    "temperatura": 18.0, "humedad": 70.0, "viento": 10.0,
    "visibilidad": 8000.0, "presion": 1015.0,
}


# =========================================================================
# Fórmulas aeronáuticas
# =========================================================================

def test_viento_cruzado_perpendicular():
    """Viento a 90° de la pista: todo es componente cruzada."""
    assert viento_cruzado(20, 90, 180) == pytest.approx(20.0)


def test_viento_cruzado_alineado():
    """Viento alineado con la pista: no hay componente cruzada."""
    assert viento_cruzado(20, 180, 180) == pytest.approx(0.0, abs=1e-9)


def test_viento_frente_positivo_y_cola_negativo():
    """De frente suma sustentación; de cola alarga el aterrizaje."""
    assert viento_frente(20, 180, 180) == pytest.approx(20.0)
    assert viento_frente(20, 0, 180) == pytest.approx(-20.0)


def test_viento_cruzado_cruza_el_norte():
    """
    Viento 350°, pista 010°: la diferencia real es 20°, no 340°.
    Un error de signo aquí convierte un viento casi alineado en uno
    perpendicular.
    """
    assert viento_cruzado(20, 350, 10) == pytest.approx(20 * math.sin(math.radians(20)))


def test_punto_rocio_con_saturacion():
    """Al 100 % de humedad, el punto de rocío iguala la temperatura."""
    assert punto_rocio(20.0, 100.0) == pytest.approx(20.0, abs=0.1)


def test_punto_rocio_siempre_menor_que_temperatura():
    for humedad in (10, 30, 50, 70, 90):
        assert punto_rocio(20.0, humedad) < 20.0


def test_altitud_densidad_sube_con_el_calor():
    """Aire caliente = menos densidad = pista efectiva más corta."""
    fria = altitud_densidad(0.0, 1013.0, 2548.0)
    caliente = altitud_densidad(30.0, 1013.0, 2548.0)
    assert caliente > fria


def test_riesgo_hielo_requiere_las_tres_condiciones():
    # Frío + spread pequeño + precipitación
    assert riesgo_hielo(5.0, 3.0, 2.0) == 1
    # Sin precipitación no hay engelamiento
    assert riesgo_hielo(5.0, 3.0, 0.0) == 0
    # Demasiado calor
    assert riesgo_hielo(25.0, 23.0, 2.0) == 0


# =========================================================================
# Completado
# =========================================================================

def test_completa_todas_las_features_base():
    df, imputados = complete_raw_features(pd.DataFrame([PAYLOAD_MINIMO]), icao="SKBO")

    requeridas = NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    faltantes = [c for c in requeridas if c not in df.columns]

    assert faltantes == []
    assert df.notna().all().all()


def test_reporta_lo_que_imputa():
    _, imputados = complete_raw_features(pd.DataFrame([PAYLOAD_MINIMO]), icao="SKBO")

    # Lo aportado por el cliente no se marca como imputado.
    for observado in PAYLOAD_MINIMO:
        assert observado not in imputados

    # Lo estimado sí.
    assert "turbulencia" in imputados
    assert "estado_pista" in imputados


def test_usa_metadata_del_aeropuerto():
    """El ICAO determina rumbo de pista y elevación reales."""
    df, _ = complete_raw_features(pd.DataFrame([PAYLOAD_MINIMO]), icao="SKBO")

    assert df["runway_heading"].iloc[0] == AIRPORTS["SKBO"]["runway_heading"]
    assert df["altitud_aeropuerto"].iloc[0] == AIRPORTS["SKBO"]["altitud"]


def test_icao_desconocido_cae_al_defecto():
    df, _ = complete_raw_features(pd.DataFrame([PAYLOAD_MINIMO]), icao="XXXX")
    assert df["altitud_aeropuerto"].iloc[0] == AIRPORTS["SKBO"]["altitud"]


def test_no_pisa_valores_observados():
    """Si el cliente aporta un dato, se respeta."""
    payload = {**PAYLOAD_MINIMO, "turbulencia": "severa", "rafagas": 55.0}
    df, imputados = complete_raw_features(pd.DataFrame([payload]), icao="SKBO")

    assert df["turbulencia"].iloc[0] == "severa"
    assert df["rafagas"].iloc[0] == 55.0
    assert "turbulencia" not in imputados


@pytest.mark.parametrize("condicion,esperado", [
    ("tormenta", "tormenta"),
    ("Niebla", "niebla"),
    ("thunderstorm", "tormenta"),
    ("parcialmente nublado", "nublado"),
    ("lluvia moderada", "lluvia_ligera"),
])
def test_mapea_condicion_libre(condicion, esperado):
    payload = {**PAYLOAD_MINIMO, "condicion": condicion}
    df, _ = complete_raw_features(pd.DataFrame([payload]), icao="SKBO")
    assert df["descripcion"].iloc[0] == esperado


def test_condicion_desconocida_no_rompe():
    payload = {**PAYLOAD_MINIMO, "condicion": "lluvia de ranas"}
    df, _ = complete_raw_features(pd.DataFrame([payload]), icao="SKBO")
    assert df["descripcion"].iloc[0] in PERFILES


def test_perfil_es_condicional_a_la_condicion():
    """
    La regresión que hizo que niebla saliera BAJO: con un default global
    se completaba techo_nubes=3000 y pista seca incluso reportando niebla,
    una combinación que el modelo nunca vio.
    """
    niebla, _ = complete_raw_features(
        pd.DataFrame([{**PAYLOAD_MINIMO, "condicion": "niebla"}]), icao="SKBO"
    )
    despejado, _ = complete_raw_features(
        pd.DataFrame([{**PAYLOAD_MINIMO, "condicion": "despejado"}]), icao="SKBO"
    )

    assert niebla["techo_nubes"].iloc[0] < despejado["techo_nubes"].iloc[0]
    assert niebla["tipo_nubes"].iloc[0] == "cubierto"
    assert despejado["estado_pista"].iloc[0] == "seca"


def test_es_noche_se_deriva_de_la_hora():
    df, _ = complete_raw_features(
        pd.DataFrame([{**PAYLOAD_MINIMO, "hora": 23}]), icao="SKBO"
    )
    assert df["es_noche"].iloc[0] == 1

    df, _ = complete_raw_features(
        pd.DataFrame([{**PAYLOAD_MINIMO, "hora": 12}]), icao="SKBO"
    )
    assert df["es_noche"].iloc[0] == 0


def test_completa_varias_filas():
    df, _ = complete_raw_features(
        pd.DataFrame([PAYLOAD_MINIMO, {**PAYLOAD_MINIMO, "temperatura": 5.0}]),
        icao="SKBO",
    )
    assert len(df) == 2


# =========================================================================
# Fallback a mock
# =========================================================================

def test_servicio_sin_modelo_marca_mock():
    """Sin modelo, el resultado debe declararse como heurístico."""
    servicio = MLServiceV2(model_path="/ruta/que/no/existe.pkl")
    resultado = servicio.predict_batch(pd.DataFrame([PAYLOAD_MINIMO]))

    assert resultado["model_status"].iloc[0] == MODEL_STATUS_MOCK
    assert resultado["mock_reason"].iloc[0]


def test_mock_no_inventa_confianza():
    """
    Regresión: el mock devolvía confianza 0.85 y probabilidades fijas,
    indistinguibles de una predicción del modelo.
    """
    servicio = MLServiceV2(model_path="/ruta/que/no/existe.pkl")
    resultado = servicio.predict_batch(pd.DataFrame([PAYLOAD_MINIMO]))

    assert resultado["confianza"].iloc[0] == 0.0
    assert pd.isna(resultado["prob_ALTO"].iloc[0])


def test_mock_sigue_ordenando_por_severidad():
    """Aun siendo heurístico, debe ser coherente."""
    servicio = MLServiceV2(model_path="/ruta/que/no/existe.pkl")

    malo = servicio.predict_batch(pd.DataFrame([
        {**PAYLOAD_MINIMO, "visibilidad": 500.0, "viento": 50.0, "humedad": 95.0}
    ]))
    bueno = servicio.predict_batch(pd.DataFrame([
        {**PAYLOAD_MINIMO, "visibilidad": 9999.0, "viento": 5.0, "humedad": 50.0}
    ]))

    assert malo["riesgo"].iloc[0] == "ALTO"
    assert bueno["riesgo"].iloc[0] == "BAJO"
