# Guardar como: backend/features/build_features.py

"""
Feature engineering para modelo de riesgo aeronáutico
Procesa 30+ variables aeronáuticas
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Variables categóricas a encodear
CATEGORICAL_FEATURES = [
    'descripcion', 'tipo_nubes', 'turbulencia', 'estado_pista'
]

# Variables numéricas a escalar
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


def build_features(raw_df: pd.DataFrame, fit=False, scaler=None, encoders=None):
    """
    Construye features para el modelo
    
    Args:
        raw_df: DataFrame con datos crudos
        fit: Si True, ajusta transformadores (solo en entrenamiento)
        scaler: StandardScaler pre-ajustado (para predicción)
        encoders: Dict de LabelEncoders pre-ajustados (para predicción)
        
    Returns:
        X: DataFrame con features procesadas
        artifacts: Dict con scaler y encoders (solo si fit=True)
    """
    df = raw_df.copy()
    
    # Si no hay encoders, crear nuevos
    if encoders is None:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                encoders[col] = LabelEncoder()
    
    # Encodear categóricas
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if fit:
                df[col] = encoders[col].fit_transform(df[col].astype(str))
            else:
                # Manejar valores no vistos
                df[col] = df[col].apply(
                    lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
                )
                df[col] = encoders[col].transform(df[col].astype(str))
    
    # Convertir booleanas a int
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Crear features derivadas
    if 'viento' in df.columns and 'rafagas' in df.columns:
        df['diferencia_rafagas'] = df['rafagas'] - df['viento']
    
    if 'temperatura' in df.columns and 'punto_rocio' in df.columns:
        df['spread_temp_dewpoint'] = df['temperatura'] - df['punto_rocio']
    
    if 'viento_cruzado' in df.columns and 'viento' in df.columns:
        df['ratio_crosswind'] = df['viento_cruzado'] / (df['viento'] + 1)
    
    # Seleccionar features finales
    feature_cols = []
    for col in NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES:
        if col in df.columns:
            feature_cols.append(col)
    
    # Agregar features derivadas
    feature_cols.extend(['diferencia_rafagas', 'spread_temp_dewpoint', 'ratio_crosswind'])
    
    X = df[feature_cols].copy()
    
    # Escalar numéricas
    numerical_cols = [c for c in X.columns if c in NUMERICAL_FEATURES or c in ['diferencia_rafagas', 'spread_temp_dewpoint', 'ratio_crosswind']]
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    if fit:
        artifacts = {
            'scaler': scaler,
            'encoders': encoders,
            'feature_names': X.columns.tolist()
        }
        return X, artifacts
    else:
        return X