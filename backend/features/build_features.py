"""
Feature engineering para el modelo de riesgo aeronautico.

Contrato: entrenamiento e inferencia DEBEN producir exactamente las mismas
29 columnas, en el mismo orden, con las mismas transformaciones. Cualquier
divergencia produce predicciones silenciosamente erroneas, que es peor que
un error.

Por eso:
  - fit=True  -> ajusta scaler y encoders, y los devuelve para persistirlos.
  - fit=False -> EXIGE el scaler y los encoders ajustados. Si no llegan,
                 lanza excepcion en vez de fabricar unos nuevos sin ajustar.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Variables categoricas a encodear
CATEGORICAL_FEATURES = [
    'descripcion', 'tipo_nubes', 'turbulencia', 'estado_pista'
]

# Variables numericas a escalar
NUMERICAL_FEATURES = [
    'temperatura', 'humedad', 'viento', 'visibilidad', 'precipitacion',
    'direccion_viento', 'runway_heading', 'viento_cruzado', 'viento_frente',
    'rafagas', 'techo_nubes', 'presion', 'altitud_aeropuerto',
    'altitud_densidad', 'punto_rocio', 'hora', 'mes', 'dia_año'
]

# Variables booleanas
BOOLEAN_FEATURES = [
    'riesgo_hielo', 'tormenta_electrica', 'cizalladura_viento', 'es_noche'
]

# Features derivadas de otras columnas
DERIVED_FEATURES = [
    'diferencia_rafagas', 'spread_temp_dewpoint', 'ratio_crosswind'
]

# Orden congelado de las 29 columnas que espera el modelo de produccion.
# Coincide con models/production/feature_names.txt. No reordenar sin
# reentrenar.
FEATURE_ORDER = (
    NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES + DERIVED_FEATURES
)

# Columnas que se escalan (numericas + derivadas)
SCALED_FEATURES = NUMERICAL_FEATURES + DERIVED_FEATURES


class FeaturePipelineError(RuntimeError):
    """El pipeline de features no puede producir la matriz esperada."""


def build_features(raw_df: pd.DataFrame, fit=False, scaler=None, encoders=None):
    """
    Construye la matriz de features para el modelo.

    Args:
        raw_df: DataFrame con las 26 columnas base (ver features/schema.py).
                Para completar un payload parcial, usar antes
                features.defaults.complete_raw_features().
        fit: True solo durante entrenamiento.
        scaler: StandardScaler ya ajustado. Obligatorio si fit=False.
        encoders: dict {columna: LabelEncoder} ya ajustados. Obligatorio
                  si fit=False.

    Returns:
        fit=False -> X (DataFrame de 29 columnas en FEATURE_ORDER)
        fit=True  -> (X, artifacts) con scaler, encoders y feature_names

    Raises:
        FeaturePipelineError: si faltan columnas base, o si se pide
            inferencia sin scaler/encoders ajustados.
    """
    df = raw_df.copy()

    if not fit:
        if scaler is None or encoders is None:
            raise FeaturePipelineError(
                "En inferencia hay que pasar el scaler y los encoders del "
                "entrenamiento. Sin ellos las features quedan en una escala "
                "distinta a la que vio el modelo y la prediccion no significa "
                "nada."
            )

    faltantes = [
        c for c in NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
        if c not in df.columns
    ]
    if faltantes:
        raise FeaturePipelineError(
            f"Faltan columnas base requeridas por el modelo: {faltantes}. "
            f"Usar features.defaults.complete_raw_features() antes de llamar "
            f"a build_features()."
        )

    # --- Categoricas -------------------------------------------------
    if fit:
        encoders = encoders or {}
        for col in CATEGORICAL_FEATURES:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
    else:
        for col in CATEGORICAL_FEATURES:
            encoder = encoders.get(col)
            if encoder is None:
                raise FeaturePipelineError(f"Falta el encoder de '{col}'")
            conocidas = set(encoder.classes_)
            # Una categoria no vista en entrenamiento se mapea a la primera
            # clase conocida. Es una decision arbitraria pero explicita:
            # el modelo no sabe nada de ese valor.
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in conocidas else encoder.classes_[0]
            )
            df[col] = encoder.transform(df[col])

    # --- Booleanas ---------------------------------------------------
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(bool).astype(int)

    # --- Derivadas ---------------------------------------------------
    # Se calculan SIEMPRE. Antes dependian de que existieran las columnas
    # fuente, asi que en inferencia se referenciaban columnas inexistentes
    # y el pipeline reventaba con KeyError.
    df['diferencia_rafagas'] = df['rafagas'] - df['viento']
    df['spread_temp_dewpoint'] = df['temperatura'] - df['punto_rocio']
    df['ratio_crosswind'] = df['viento_cruzado'] / (df['viento'] + 1)

    # --- Seleccion y orden -------------------------------------------
    X = df[FEATURE_ORDER].copy()

    # --- Escalado ----------------------------------------------------
    if fit:
        scaler = scaler or StandardScaler()
        X[SCALED_FEATURES] = scaler.fit_transform(X[SCALED_FEATURES])
    else:
        X[SCALED_FEATURES] = scaler.transform(X[SCALED_FEATURES])

    if fit:
        artifacts = {
            'scaler': scaler,
            'encoders': encoders,
            'feature_names': list(X.columns),
        }
        return X, artifacts

    return X
