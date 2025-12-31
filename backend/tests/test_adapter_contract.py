import pandas as pd
from features.build_features import build_features


def test_api_payload_to_features():
    payload = {
        "temperatura_c": 20.0,
        "humedad_pct": 75.0,
        "presion_hpa": 1010.0,
        "velocidad_viento_ms": 7.0,
        "rafaga_viento_ms": 10.0,
        "visibilidad_m": 6000.0,
        "precipitacion_mm": 0.0,
        "nubes_pct": 50.0,
        "riesgo_hielo": 0,
    }

    df = pd.DataFrame([payload])
    X = build_features(df)

    assert X.shape == (1, len(payload))
