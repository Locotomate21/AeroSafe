"""
Features del modelo de pronostico.

Punto unico de verdad para las 21 features que consume el pronostico, de
modo que el constructor del dataset (entrenamiento) y el servicio de la
API (inferencia) las calculen EXACTAMENTE igual. Duplicar esta logica es
como se genero el desajuste train/serve que costo semanas de
predicciones falsas en el modelo anterior; no se repite.

Cada feature se calcula a partir de UNA sola observacion (el METAR
actual), sin mirar el futuro: son las variables disponibles en el
instante t para pronosticar t+N.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.wx_codes import intensidad_precipitacion

CONDICIONES_ADVERSAS = ("niebla", "tormenta")

# Variables base observables en t (no incluye 'descripcion': es de donde
# sale la etiqueta, meterla como feature reintroduce circularidad).
FEATURES_BASE = [
    "temperatura", "punto_rocio", "humedad", "viento", "rafagas",
    "direccion_viento", "visibilidad", "presion", "techo_nubes",
    "viento_cruzado", "altitud_densidad",
    "hora", "mes", "es_noche",
]

# Features derivadas anadidas por add_forecast_features().
FEATURES_DERIVADAS = [
    "precip_intensidad", "adverso_actual",
    "hora_sin", "hora_cos", "mes_sin", "mes_cos", "spread_t_td",
]

# Orden congelado de las 21 features que espera el modelo. Coincide con
# models/forecast/features_<icao>_h<N>.txt.
FORECAST_FEATURES = FEATURES_BASE + FEATURES_DERIVADAS


def es_adverso(descripcion: pd.Series) -> pd.Series:
    """1 si la condicion actual es niebla o tormenta."""
    return descripcion.isin(CONDICIONES_ADVERSAS).astype(int)


def add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anade las 7 features derivadas a un DataFrame que ya trae las base y
    las columnas 'descripcion' y 'metar'.

    No empareja con el futuro ni calcula la etiqueta: solo deriva las
    features del instante actual. Lo usan por igual el constructor del
    dataset y el servicio de inferencia.
    """
    df = df.copy()

    # Precipitacion como escala ordinal desde el METAR crudo (la columna
    # numerica del IEM es cero para SKBO).
    df["precip_intensidad"] = df["metar"].apply(intensidad_precipitacion)

    # Persistencia: la condicion actual es la senal mas fuerte a corto
    # plazo, y se le da explicitamente al modelo.
    df["adverso_actual"] = es_adverso(df["descripcion"])

    # Ciclicas: hora y mes son circulares (las 23h estan al lado de las 0h).
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    # Spread temperatura-rocio: proxy de saturacion; cerca de cero, el
    # aire esta saturado y la niebla es inminente.
    df["spread_t_td"] = df["temperatura"] - df["punto_rocio"]

    return df
