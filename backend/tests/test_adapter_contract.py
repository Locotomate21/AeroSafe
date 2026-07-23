"""
Contrato de extremo a extremo: JSON de OpenWeather -> matriz de features.

Es el camino que recorre una petición real, así que se prueba entero.
"""
import pytest

from features.adapters.openweather_adapter import openweather_to_raw_df
from features.build_features import FEATURE_ORDER, build_features
from features.defaults import complete_raw_features


PAYLOAD_OPENWEATHER = {
    "weather": [{"main": "Rain", "description": "lluvia ligera"}],
    "main": {"temp": 22.0, "humidity": 60.0, "pressure": 1013.0},
    "wind": {"speed": 5.0, "deg": 90, "gust": 8.0},
    "visibility": 8000,
    "rain": {"1h": 1.2},
    "clouds": {"all": 40},
}


def test_adapter_produce_nombres_canonicos():
    """El adaptador traduce al vocabulario del schema, no al de la API."""
    df = openweather_to_raw_df(PAYLOAD_OPENWEATHER)

    for col in ("temperatura", "humedad", "presion", "viento", "visibilidad"):
        assert col in df.columns

    assert df["temperatura"].iloc[0] == 22.0
    assert df["humedad"].iloc[0] == 60.0
    assert df["visibilidad"].iloc[0] == 8000.0


def test_adapter_convierte_viento_a_kmh():
    """
    OpenWeather reporta m/s y el modelo se entrenó con km/h.

    Sin conversión, 5 m/s (18 km/h) entraría como 5 km/h y el modelo vería
    un viento cuatro veces más flojo del real.
    """
    df = openweather_to_raw_df(PAYLOAD_OPENWEATHER)

    assert df["viento"].iloc[0] == pytest.approx(5.0 * 3.6)
    assert df["rafagas"].iloc[0] == pytest.approx(8.0 * 3.6)


def test_adapter_mapea_condicion():
    df = openweather_to_raw_df(PAYLOAD_OPENWEATHER)
    assert df["descripcion"].iloc[0] == "lluvia_ligera"


def test_adapter_suma_lluvia_y_nieve():
    payload = dict(PAYLOAD_OPENWEATHER, rain={"1h": 1.5}, snow={"1h": 0.5})
    df = openweather_to_raw_df(payload)
    assert df["precipitacion"].iloc[0] == pytest.approx(2.0)


def test_payload_invalido_da_error_claro():
    with pytest.raises(ValueError, match="OpenWeather"):
        openweather_to_raw_df({"main": {}})


def test_recorrido_completo_hasta_features(pipeline_ajustado):
    """OpenWeather -> adaptador -> completado -> 29 features en orden."""
    raw = openweather_to_raw_df(PAYLOAD_OPENWEATHER)
    completo, imputados = complete_raw_features(raw, icao="SKBO")

    X = build_features(
        completo,
        scaler=pipeline_ajustado["scaler"],
        encoders=pipeline_ajustado["encoders"],
    )

    assert list(X.columns) == FEATURE_ORDER
    assert X.notna().all().all()
    # El payload de OpenWeather no trae pista, turbulencia ni datos
    # temporales: deben quedar registrados como imputados.
    assert "estado_pista" in imputados
    assert "turbulencia" in imputados
