from typing import Dict, Tuple
import pandas as pd


# =========================
# Versión del esquema
# =========================
SCHEMA_VERSION = "1.0.0"


# =========================
# Definición de Features
# nombre: (tipo, min, max)
# =========================
FEATURES: Dict[str, Tuple[type, float | None, float | None]] = {
    "temperatura_c": (float, -60.0, 60.0),
    "humedad_pct": (float, 0.0, 100.0),
    "presion_hpa": (float, 850.0, 1100.0),
    "velocidad_viento_ms": (float, 0.0, 80.0),
    "rafaga_viento_ms": (float, 0.0, 120.0),
    "visibilidad_m": (float, 0.0, 100_000.0),
    "precipitacion_mm": (float, 0.0, 500.0),
    "nubes_pct": (float, 0.0, 100.0),
    "riesgo_hielo": (int, 0, 1),
}


# =========================
# Orden congelado de features
# =========================
FEATURE_ORDER = list(FEATURES.keys())


# =========================
# Target
# =========================
TARGET = "riesgo_operacional"


# =========================
# Columnas totales (útil para training)
# =========================
ALL_COLUMNS = FEATURE_ORDER + [TARGET]


# =========================
# Validación del DataFrame
# =========================
def validate_dataframe(
    df: pd.DataFrame,
    *,
    check_ranges: bool = True,
) -> None:
    """
    Valida que el DataFrame cumpla el contrato del schema.

    - Siempre valida estructura, orden y tipos.
    - La validación de rangos puede desactivarse (útil para tests de contrato).
    """

    # --- Validar columnas faltantes ---
    missing_cols = set(FEATURE_ORDER) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Features faltantes: {missing_cols}")

    # --- Validar columnas extra ---
    extra_cols = set(df.columns) - set(FEATURE_ORDER)
    if extra_cols:
        raise ValueError(f"Columnas no esperadas: {extra_cols}")

    # --- Validar orden de columnas ---
    if list(df.columns) != FEATURE_ORDER:
        raise ValueError(
            f"Orden de columnas inválido. "
            f"Esperado: {FEATURE_ORDER}, "
            f"Recibido: {list(df.columns)}"
        )

    # --- Validar tipos y rangos ---
    for col, (dtype, min_v, max_v) in FEATURES.items():
        series = df[col]

        # Validación de tipo (compatible con pandas / numpy)
        if dtype is float:
            if not pd.api.types.is_numeric_dtype(series):
                raise TypeError(f"Tipo inválido en {col}, se esperaba float")
        elif dtype is int:
            if not pd.api.types.is_integer_dtype(series):
                raise TypeError(f"Tipo inválido en {col}, se esperaba int")

        # Validación de rango (opcional)
        if check_ranges:
            if min_v is not None and (series.dropna() < min_v).any():
                raise ValueError(f"Valor fuera de rango mínimo en {col}")

            if max_v is not None and (series.dropna() > max_v).any():
                raise ValueError(f"Valor fuera de rango máximo en {col}")
