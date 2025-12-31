from features.build_features import build_features
from features.schema import FEATURES


def test_build_features_schema_and_order(sample_raw_weather):
    X = build_features(sample_raw_weather)

    # Columnas y orden congelado
    assert list(X.columns) == list(FEATURES.keys())

    # Tipos correctos
    for col, (dtype, _, _) in FEATURES.items():
        assert X[col].dtype == dtype
