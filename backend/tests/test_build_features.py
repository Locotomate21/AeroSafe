from backend.features.build_features import build_features  # ← CORREGIDO


def test_build_features_schema_and_order(sample_raw_weather):
    """
    Test que build_features respeta el schema y orden definidos.
    """
    features = build_features(sample_raw_weather)

    # Validar que las columnas están en el orden correcto
    from backend.features.schema import FEATURE_ORDER  # ← CORREGIDO
    assert list(features.columns) == FEATURE_ORDER

    # Validar tipos
    assert features["temperatura_c"].dtype in ["float64", "float32"]
    assert features["humedad_pct"].dtype in ["float64", "float32"]
    assert features["riesgo_hielo"].dtype in ["int64", "int32"]
