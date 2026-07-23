"""
Contrato del pipeline de features.

Lo que se protege aquí: que entrenamiento e inferencia produzcan las
mismas 29 columnas en el mismo orden, y que sea imposible inferir sin los
transformadores ajustados.
"""
import pandas as pd
import pytest

from features.build_features import (
    FEATURE_ORDER,
    FeaturePipelineError,
    build_features,
)


def test_orden_y_numero_de_columnas(fila_completa, pipeline_ajustado):
    """La matriz de features respeta el orden congelado."""
    X = build_features(
        fila_completa,
        scaler=pipeline_ajustado["scaler"],
        encoders=pipeline_ajustado["encoders"],
    )

    assert list(X.columns) == FEATURE_ORDER
    assert len(X.columns) == 29


def test_features_derivadas_siempre_presentes(fila_completa, pipeline_ajustado):
    """
    Las tres derivadas se calculan siempre.

    Regresión: antes solo se creaban si existían las columnas fuente, así
    que en inferencia se seleccionaban columnas inexistentes y el pipeline
    reventaba con KeyError, que el servicio tragaba cayendo a mock.
    """
    X = build_features(
        fila_completa,
        scaler=pipeline_ajustado["scaler"],
        encoders=pipeline_ajustado["encoders"],
    )

    for col in ("diferencia_rafagas", "spread_temp_dewpoint", "ratio_crosswind"):
        assert col in X.columns
        assert X[col].notna().all()


def test_inferencia_sin_scaler_falla(fila_completa):
    """
    Sin scaler ajustado hay que fallar, no improvisar.

    Regresión: build_features creaba un StandardScaler nuevo sin ajustar,
    dejando las features en una escala que el modelo nunca vio.
    """
    with pytest.raises(FeaturePipelineError, match="scaler"):
        build_features(fila_completa)


def test_inferencia_sin_encoders_falla(fila_completa, pipeline_ajustado):
    with pytest.raises(FeaturePipelineError):
        build_features(fila_completa, scaler=pipeline_ajustado["scaler"])


def test_columnas_base_faltantes_dan_error_claro(pipeline_ajustado):
    """Un payload incompleto produce un error que nombra lo que falta."""
    incompleto = pd.DataFrame([{"temperatura": 20.0, "humedad": 60.0}])

    with pytest.raises(FeaturePipelineError, match="Faltan columnas base"):
        build_features(
            incompleto,
            scaler=pipeline_ajustado["scaler"],
            encoders=pipeline_ajustado["encoders"],
        )


def test_categoria_no_vista_no_rompe(fila_completa, pipeline_ajustado):
    """Una categoría desconocida se mapea a una conocida en vez de fallar."""
    df = fila_completa.copy()
    df["descripcion"] = "lluvia_de_ranas"

    X = build_features(
        df,
        scaler=pipeline_ajustado["scaler"],
        encoders=pipeline_ajustado["encoders"],
    )

    assert len(X) == 1
    assert X["descripcion"].notna().all()


def test_fit_devuelve_artefactos(fila_completa):
    """En entrenamiento se devuelven scaler, encoders y nombres."""
    X, artifacts = build_features(fila_completa, fit=True)

    assert set(artifacts) == {"scaler", "encoders", "feature_names"}
    assert artifacts["feature_names"] == FEATURE_ORDER
    assert list(X.columns) == FEATURE_ORDER
