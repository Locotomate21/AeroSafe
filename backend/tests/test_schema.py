import pandas as pd
import pytest

from features.schema import FEATURES, validate_dataframe


def test_validate_dataframe_ok():
    # Usa el valor mínimo válido de cada feature
    data = {}
    for col, (dtype, min_v, max_v) in FEATURES.items():
        if dtype is float:
            # Si hay mínimo, usa ese; si no, usa 1.0
            data[col] = min_v if min_v is not None else 1.0
        else:  # int
            # Si hay mínimo, usa ese; si no, usa 1
            data[col] = min_v if min_v is not None else 1

    df = pd.DataFrame([data])
    validate_dataframe(df)


def test_missing_feature_raises():
    data = {
        col: 1.0
        for col in list(FEATURES.keys())[:-1]
    }

    df = pd.DataFrame([data])

    with pytest.raises(ValueError):
        validate_dataframe(df)