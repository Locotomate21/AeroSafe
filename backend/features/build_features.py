import pandas as pd
import numpy as np

from backend.features.schema import FEATURES, FEATURE_ORDER, validate_dataframe


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte datos crudos (API / CSV / batch)
    en features finales listas para el modelo.
    """

    df = raw_df.copy()

    # =========================
    # Renombrar columnas externas → internas
    # =========================
    # Mapeo desde inglés (API externa)
    rename_map_en = {
        "temp": "temperatura_c",
        "humidity": "humedad_pct",
        "pressure": "presion_hpa",
        "wind_speed": "velocidad_viento_ms",
        "wind_gust": "rafaga_viento_ms",
        "visibility": "visibilidad_m",
        "precipitation": "precipitacion_mm",
        "clouds": "nubes_pct",
        "ice_risk": "riesgo_hielo",
    }
    
    # Mapeo desde español (tests o API local)
    rename_map_es = {
        "temperatura": "temperatura_c",
        "humedad": "humedad_pct",
        "presion": "presion_hpa",
        "viento": "velocidad_viento_ms",
        "rafaga": "rafaga_viento_ms",
        "visibilidad": "visibilidad_m",
        "precipitacion": "precipitacion_mm",
        "nubes": "nubes_pct",
        "hielo": "riesgo_hielo",
    }

    # Aplicar ambos mapeos (solo renombra las que existan)
    df = df.rename(columns={**rename_map_en, **rename_map_es})

    # =========================
    # Validar columnas esperadas
    # =========================
    missing = set(FEATURE_ORDER) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes tras adaptación: {missing}")

    # =========================
    # Seleccionar features (orden congelado)
    # =========================
    df = df[FEATURE_ORDER]

    # =========================
    # Tipado explícito
    # =========================
    for col, (dtype, _, _) in FEATURES.items():
        if dtype is float:
            df[col] = pd.to_numeric(df[col], errors="raise")
        elif dtype is int:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    # =========================
    # Validación final del contrato
    # =========================
    validate_dataframe(df)

    return df