import pandas as pd
from backend.features.build_features import build_features  # ← CORREGIDO


def test_api_payload_to_features():
    """
    Test que el payload de API se transforma correctamente en features.
    """
    # Payload simulado de OpenWeather API
    api_payload = {
        "temp": 22.0,
        "humidity": 60.0,
        "pressure": 1013.0,
        "wind_speed": 5.0,
        "wind_gust": 8.0,
        "visibility": 8000.0,
        "precipitation": 0.0,
        "clouds": 40.0,
        "ice_risk": 0,
    }

    raw_df = pd.DataFrame([api_payload])
    features = build_features(raw_df)

    # Validar que las columnas se transformaron correctamente
    assert "temperatura_c" in features.columns
    assert "humedad_pct" in features.columns
    assert "presion_hpa" in features.columns
    assert "velocidad_viento_ms" in features.columns

    # Validar valores
    assert features["temperatura_c"].iloc[0] == 22.0
    assert features["humedad_pct"].iloc[0] == 60.0