"""
Completado de features para inferencia.

El modelo de produccion espera 26 variables base (mas 3 derivadas), pero la
API expone un payload reducido: un cliente tipicamente solo conoce
temperatura, humedad, viento, visibilidad y presion.

Este modulo rellena el resto. Dos reglas que no se negocian:

1. Lo que se pueda *derivar* se deriva con las mismas formulas que uso el
   generador del dataset (ml/scripts/generate_dataset_UNIFIED.py). Si
   entrenamiento e inferencia calculan viento_cruzado distinto, el modelo
   recibe una distribucion que nunca vio.

2. Lo que no se pueda derivar se imputa con un valor documentado, y el
   nombre del campo imputado se devuelve al llamador. Una prediccion
   basada en 15 valores inventados no vale lo mismo que una basada en
   datos reales, y quien consume la API tiene derecho a saberlo.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

# =========================================================================
# Metadata de aeropuertos
# =========================================================================
# runway_heading: rumbo magnetico de la pista principal en grados.
# altitud: elevacion del aerodromo en metros.

AIRPORTS: Dict[str, Dict[str, float]] = {
    "SKBO": {"runway_heading": 134.0, "altitud": 2548.0},  # Bogota / El Dorado
    "SKRG": {"runway_heading": 182.0, "altitud": 2142.0},  # Rionegro / JMC
    "SKCL": {"runway_heading": 20.0, "altitud": 964.0},    # Cali / Bonilla Aragon
    "SKBQ": {"runway_heading": 50.0, "altitud": 30.0},     # Barranquilla / Cortissoz
    "SKCG": {"runway_heading": 19.0, "altitud": 4.0},      # Cartagena / Nunez
}

# Aeropuerto por defecto cuando no se indica ICAO. AeroSafe se desarrolla
# contra SKBO, asi que asumir SKBO es menos malo que asumir nivel del mar.
DEFAULT_AIRPORT = "SKBO"

# Perfiles por condicion meteorologica.
#
# Un default global tipo "dia despejado" para todas las variables no
# observadas es activamente daNino: en el dataset, descripcion esta muy
# correlacionada con techo_nubes, tipo_nubes, turbulencia y estado_pista.
# Reportar 'niebla' con techo de 3000 m y pista seca es una combinacion que
# el modelo no vio nunca, y las 20 senales benignas ahogan a la unica mala.
#
# Estos valores son la mediana (numericas) y la moda (categoricas) por
# descripcion en data/dataset/weather_risk_aviation.csv. Es decir, la mejor
# estimacion dado lo unico que sabemos. Recalcular si se regenera el dataset.
PERFILES: Dict[str, Dict[str, Any]] = {
    "despejado": {
        "techo_nubes": 9898.0, "tipo_nubes": "despejado", "turbulencia": "ninguna",
        "estado_pista": "seca", "tormenta_electrica": 0, "cizalladura_viento": 0,
        "precipitacion": 0.0,
    },
    "nublado": {
        "techo_nubes": 2722.0, "tipo_nubes": "dispersas", "turbulencia": "leve",
        "estado_pista": "humeda", "tormenta_electrica": 0, "cizalladura_viento": 0,
        "precipitacion": 1.0,
    },
    "lluvia_ligera": {
        "techo_nubes": 1666.0, "tipo_nubes": "nublado", "turbulencia": "leve",
        "estado_pista": "humeda", "tormenta_electrica": 0, "cizalladura_viento": 0,
        "precipitacion": 4.9,
    },
    "lluvia_fuerte": {
        "techo_nubes": 865.0, "tipo_nubes": "cubierto", "turbulencia": "moderada",
        "estado_pista": "mojada", "tormenta_electrica": 0, "cizalladura_viento": 0,
        "precipitacion": 17.6,
    },
    "tormenta": {
        "techo_nubes": 610.0, "tipo_nubes": "cubierto", "turbulencia": "severa",
        "estado_pista": "contaminada", "tormenta_electrica": 1, "cizalladura_viento": 0,
        "precipitacion": 26.1,
    },
    "niebla": {
        "techo_nubes": 494.0, "tipo_nubes": "cubierto", "turbulencia": "ninguna",
        "estado_pista": "humeda", "tormenta_electrica": 0, "cizalladura_viento": 0,
        "precipitacion": 1.0,
    },
    "nieve": {
        "techo_nubes": 1265.0, "tipo_nubes": "cubierto", "turbulencia": "leve",
        "estado_pista": "contaminada", "tormenta_electrica": 0, "cizalladura_viento": 0,
        "precipitacion": 11.9,
    },
    "granizo": {
        "techo_nubes": 1284.0, "tipo_nubes": "cubierto", "turbulencia": "severa",
        "estado_pista": "contaminada", "tormenta_electrica": 1, "cizalladura_viento": 0,
        "precipitacion": 19.6,
    },
}

PERFIL_DEFECTO = "despejado"

# Columnas cubiertas por los perfiles.
COLUMNAS_PERFIL = list(PERFILES[PERFIL_DEFECTO].keys())

# Defaults que no dependen de la condicion meteorologica.
STATIC_DEFAULTS: Dict[str, Any] = {
    "direccion_viento": 0.0,
}

# Mapeo de condiciones de la API a las categorias del dataset.
CONDICION_A_DESCRIPCION = {
    "despejado": "despejado", "clear": "despejado", "soleado": "despejado",
    "nublado": "nublado", "clouds": "nublado", "nubes": "nublado",
    "parcialmente nublado": "nublado", "cubierto": "nublado",
    "lluvia": "lluvia_ligera", "rain": "lluvia_ligera",
    "lluvia ligera": "lluvia_ligera", "llovizna": "lluvia_ligera",
    "lluvia fuerte": "lluvia_fuerte", "aguacero": "lluvia_fuerte",
    "tormenta": "tormenta", "thunderstorm": "tormenta",
    "niebla": "niebla", "fog": "niebla", "mist": "niebla", "neblina": "niebla",
    "nieve": "nieve", "snow": "nieve",
    "granizo": "granizo", "hail": "granizo",
}


# =========================================================================
# Formulas aeronauticas
# =========================================================================
# Replican calculate_crosswind / calculate_headwind / calculate_density_altitude
# de ml/scripts/generate_dataset_UNIFIED.py. No cambiar una sin la otra.

def _angulo_relativo(wind_dir: float, runway_heading: float) -> float:
    """Angulo agudo entre el viento y el eje de pista, en grados."""
    diff = abs(wind_dir - runway_heading)
    if diff > 180:
        diff = 360 - diff
    return diff


def viento_cruzado(wind_speed: float, wind_dir: float, runway_heading: float) -> float:
    """Componente perpendicular al eje de pista (siempre positiva)."""
    return abs(wind_speed * math.sin(math.radians(_angulo_relativo(wind_dir, runway_heading))))


def viento_frente(wind_speed: float, wind_dir: float, runway_heading: float) -> float:
    """Componente longitudinal. Negativa = viento de cola."""
    return wind_speed * math.cos(math.radians(_angulo_relativo(wind_dir, runway_heading)))


def altitud_densidad(temp: float, presion: float, altitud: float) -> float:
    """
    Altitud de densidad en metros.

    Usa la aproximacion del generador del dataset: temperatura ISA estandar
    de 15 C con lapse rate de 2 C por cada 1000 m, y 120 m de altitud de
    densidad por cada grado de desviacion.

    Nota: esta aproximacion ignora la presion, igual que en entrenamiento.
    El parametro se mantiene en la firma para no romper el contrato cuando
    se corrija junto con un reentrenamiento.
    """
    std_temp = 15 - (altitud / 1000 * 2)
    return altitud + (120 * (temp - std_temp))


def punto_rocio(temp: float, humedad: float) -> float:
    """
    Punto de rocio por la formula de Magnus-Tetens.

    Ojo: el dataset sintetico genera el punto de rocio con ruido aleatorio
    por tramos de humedad, no con esta formula. Aqui se usa la fisica real
    porque es lo correcto para datos de verdad, pero es una fuente conocida
    de desviacion entre entrenamiento y produccion. Se resuelve regenerando
    el dataset con Magnus.
    """
    humedad = min(max(humedad, 1.0), 100.0)
    b, c = 17.625, 243.04
    gamma = math.log(humedad / 100.0) + (b * temp) / (c + temp)
    return (c * gamma) / (b - gamma)


def riesgo_hielo(temp: float, dewpoint: float, precipitacion: float) -> int:
    """Condiciones de engelamiento, mismo criterio que el generador."""
    return int(0 <= temp <= 10 and (temp - dewpoint) < 3 and precipitacion > 0)


# =========================================================================
# Completado
# =========================================================================

def complete_raw_features(
    raw_df: pd.DataFrame,
    *,
    icao: str | None = None,
    momento: datetime | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Completa un DataFrame parcial hasta las 26 features base del modelo.

    Args:
        raw_df: DataFrame con las columnas que el cliente si conoce.
        icao: Codigo ICAO, para tomar runway_heading y altitud reales.
        momento: Instante de la observacion. Por defecto, ahora.

    Returns:
        (df_completo, campos_imputados)
        campos_imputados lista las columnas que NO venian en raw_df y que
        por tanto son estimaciones, no observaciones.
    """
    df = raw_df.copy()
    presentes = set(df.columns)
    imputados: List[str] = []

    def falta(col: str) -> bool:
        return col not in df.columns or df[col].isna().all()

    # --- Aeropuerto -------------------------------------------------
    meta = AIRPORTS.get((icao or DEFAULT_AIRPORT).upper(), AIRPORTS[DEFAULT_AIRPORT])

    if falta("runway_heading"):
        df["runway_heading"] = meta["runway_heading"]
    if falta("altitud_aeropuerto"):
        df["altitud_aeropuerto"] = meta["altitud"]

    # --- Temporales -------------------------------------------------
    momento = momento or datetime.now()
    if falta("hora"):
        df["hora"] = momento.hour
    if falta("dia_año"):
        df["dia_año"] = momento.timetuple().tm_yday
    if falta("mes"):
        # El generador deriva mes de dia_año, no del calendario. Se replica
        # para que el modelo vea la misma relacion entre ambas columnas.
        df["mes"] = df["dia_año"].apply(lambda d: min(int(d) // 30 + 1, 12))
    if falta("es_noche"):
        df["es_noche"] = df["hora"].apply(lambda h: int(h < 6 or h > 20))

    # --- Condicion meteorologica ------------------------------------
    # 'condicion' es el campo libre que expone la API; se traduce a la
    # categoria del dataset antes de caer al default.
    if falta("descripcion"):
        if "condicion" in df.columns:
            df["descripcion"] = df["condicion"].apply(_mapear_condicion)
        else:
            df["descripcion"] = PERFIL_DEFECTO

    # --- Perfil segun la condicion ----------------------------------
    # Las variables no observadas se completan con el perfil tipico de la
    # condicion reportada, no con un default global benigno.
    for col in COLUMNAS_PERFIL:
        if falta(col):
            df[col] = df["descripcion"].apply(
                lambda d: PERFILES.get(str(d), PERFILES[PERFIL_DEFECTO])[col]
            )

    # --- Defaults estaticos -----------------------------------------
    for col, valor in STATIC_DEFAULTS.items():
        if falta(col):
            df[col] = valor

    # --- Derivadas de otras columnas --------------------------------
    if falta("rafagas"):
        # Factor de rafaga medio del dataset (uniforme 1.1-1.5).
        df["rafagas"] = df["viento"] * 1.3

    if falta("punto_rocio"):
        df["punto_rocio"] = df.apply(
            lambda r: punto_rocio(r["temperatura"], r["humedad"]), axis=1
        )

    if falta("viento_cruzado"):
        df["viento_cruzado"] = df.apply(
            lambda r: viento_cruzado(
                r["viento"], r["direccion_viento"], r["runway_heading"]
            ),
            axis=1,
        )

    if falta("viento_frente"):
        df["viento_frente"] = df.apply(
            lambda r: viento_frente(
                r["viento"], r["direccion_viento"], r["runway_heading"]
            ),
            axis=1,
        )

    if falta("altitud_densidad"):
        df["altitud_densidad"] = df.apply(
            lambda r: altitud_densidad(
                r["temperatura"], r["presion"], r["altitud_aeropuerto"]
            ),
            axis=1,
        )

    if falta("riesgo_hielo"):
        df["riesgo_hielo"] = df.apply(
            lambda r: riesgo_hielo(
                r["temperatura"], r["punto_rocio"], r["precipitacion"]
            ),
            axis=1,
        )

    # Todo lo que no venia en la entrada original es una estimacion.
    imputados = sorted(set(df.columns) - presentes)

    return df, imputados


def _mapear_condicion(condicion: Any) -> str:
    """Traduce el texto libre de 'condicion' a una categoria del dataset."""
    if not isinstance(condicion, str):
        return STATIC_DEFAULTS["descripcion"]
    clave = condicion.strip().lower()
    if clave in CONDICION_A_DESCRIPCION:
        return CONDICION_A_DESCRIPCION[clave]
    # Coincidencia parcial: "lluvia moderada" -> "lluvia"
    for termino, categoria in CONDICION_A_DESCRIPCION.items():
        if termino in clave:
            return categoria
    return STATIC_DEFAULTS["descripcion"]
