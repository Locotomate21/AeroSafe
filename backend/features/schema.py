"""
Schema de features para modelo de riesgo aeronáutico AeroSafe
Versión unificada con 30+ variables aeronáuticas
"""
from typing import Dict, Tuple
import pandas as pd

# =========================
# Versión del esquema
# =========================
SCHEMA_VERSION = "2.0.0"  # ⬆️ Actualizado para 30+ features

# =========================
# Definición de Features
# nombre: (tipo, min, max)
# =========================
FEATURES: Dict[str, Tuple[type, float | None, float | None]] = {
    # Variables meteorológicas básicas
    "temperatura": (float, -60.0, 60.0),
    "humedad": (float, 0.0, 100.0),
    "presion": (float, 850.0, 1100.0),
    "viento": (float, 0.0, 150.0),
    "visibilidad": (float, 0.0, 20000.0),
    "precipitacion": (float, 0.0, 500.0),
    
    # Variables de viento avanzadas
    "direccion_viento": (float, 0.0, 359.0),
    "runway_heading": (float, 0.0, 359.0),
    "viento_cruzado": (float, 0.0, 150.0),
    "viento_frente": (float, -150.0, 150.0),  # Puede ser negativo (tailwind)
    "rafagas": (float, 0.0, 200.0),
    
    # Variables de nubes y visibilidad
    "techo_nubes": (float, 0.0, 20000.0),
    "tipo_nubes": (int, 0, 3),  # Encoded: 0=despejado, 1=dispersas, 2=nublado, 3=cubierto
    
    # Variables de pista y aeropuerto
    "estado_pista": (int, 0, 4),  # Encoded: 0=seca, 1=humeda, 2=mojada, 3=contaminada, 4=nevada
    "altitud_aeropuerto": (float, 0.0, 5000.0),
    "altitud_densidad": (float, -2000.0, 15000.0),
    
    # Variables meteorológicas avanzadas
    "punto_rocio": (float, -60.0, 40.0),
    "descripcion": (int, 0, 7),  # Encoded: despejado, nublado, lluvia_ligera, etc.
    
    # Variables de turbulencia y fenómenos
    "turbulencia": (int, 0, 3),  # Encoded: 0=ninguna, 1=leve, 2=moderada, 3=severa
    
    # Variables booleanas (0 o 1)
    "riesgo_hielo": (int, 0, 1),
    "tormenta_electrica": (int, 0, 1),
    "cizalladura_viento": (int, 0, 1),
    "es_noche": (int, 0, 1),
    
    # Variables temporales
    "hora": (int, 0, 23),
    "mes": (int, 1, 12),
    "dia_año": (int, 1, 365),
    
    # Features derivadas
    "diferencia_rafagas": (float, 0.0, 100.0),
    "spread_temp_dewpoint": (float, 0.0, 50.0),
    "ratio_crosswind": (float, 0.0, 2.0),
}

# =========================
# Orden congelado de features
# =========================
FEATURE_ORDER = list(FEATURES.keys())

# =========================
# Target (3 clases unificadas)
# =========================
TARGET = "riesgo"
TARGET_CLASSES = ["BAJO", "MODERADO", "ALTO"]

# =========================
# Columnas totales
# =========================
ALL_COLUMNS = FEATURE_ORDER + [TARGET]

# =========================
# Mapeo de features categóricas
# =========================
CATEGORICAL_MAPPINGS = {
    "descripcion": ["despejado", "nublado", "lluvia_ligera", "lluvia_fuerte", 
                    "tormenta", "niebla", "nieve", "granizo"],
    "tipo_nubes": ["despejado", "dispersas", "nublado", "cubierto"],
    "estado_pista": ["seca", "humeda", "mojada", "contaminada", "nevada"],
    "turbulencia": ["ninguna", "leve", "moderada", "severa"],
}

# =========================
# Validación del DataFrame
# =========================
def validate_dataframe(
    df: pd.DataFrame,
    *,
    check_ranges: bool = True,
    strict_columns: bool = True,
) -> None:
    """
    Valida que el DataFrame cumpla el contrato del schema.
    
    Args:
        df: DataFrame a validar
        check_ranges: Si True, valida rangos de valores
        strict_columns: Si True, requiere orden exacto de columnas
    
    Raises:
        ValueError: Si hay columnas faltantes, extra, o fuera de rango
        TypeError: Si los tipos no coinciden
    """
    
    # Validar columnas faltantes
    missing_cols = set(FEATURE_ORDER) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Features faltantes: {missing_cols}")
    
    # Validar columnas extra (warning, no error crítico)
    extra_cols = set(df.columns) - set(FEATURE_ORDER)
    if extra_cols and strict_columns:
        raise ValueError(f"Columnas no esperadas: {extra_cols}")
    
    # Validar orden (solo si strict)
    if strict_columns and list(df.columns) != FEATURE_ORDER:
        raise ValueError(
            f"Orden de columnas inválido.\n"
            f"Esperado: {FEATURE_ORDER[:5]}...\n"
            f"Recibido: {list(df.columns)[:5]}..."
        )
    
    # Validar tipos y rangos
    for col, (dtype, min_v, max_v) in FEATURES.items():
        if col not in df.columns:
            continue
            
        series = df[col]
        
        # Validación de tipo
        if dtype is float:
            if not pd.api.types.is_numeric_dtype(series):
                raise TypeError(f"Tipo inválido en '{col}', se esperaba float")
        elif dtype is int:
            if not pd.api.types.is_integer_dtype(series):
                raise TypeError(f"Tipo inválido en '{col}', se esperaba int")
        
        # Validación de rango
        if check_ranges:
            if min_v is not None:
                invalid_min = (series.dropna() < min_v).any()
                if invalid_min:
                    actual_min = series.min()
                    raise ValueError(
                        f"Valor fuera de rango mínimo en '{col}': "
                        f"{actual_min} < {min_v}"
                    )
            
            if max_v is not None:
                invalid_max = (series.dropna() > max_v).any()
                if invalid_max:
                    actual_max = series.max()
                    raise ValueError(
                        f"Valor fuera de rango máximo en '{col}': "
                        f"{actual_max} > {max_v}"
                    )


def get_feature_info() -> Dict[str, Dict]:
    """
    Retorna información completa de features
    
    Returns:
        Dict con info de cada feature
    """
    info = {}
    for col, (dtype, min_v, max_v) in FEATURES.items():
        info[col] = {
            "type": dtype.__name__,
            "min": min_v,
            "max": max_v,
            "categorical": col in CATEGORICAL_MAPPINGS,
            "mapping": CATEGORICAL_MAPPINGS.get(col)
        }
    return info


def print_schema_summary():
    """Imprime resumen del schema"""
    print("=" * 70)
    print(f"AEROSAFE SCHEMA v{SCHEMA_VERSION}")
    print("=" * 70)
    print(f"\nTotal features: {len(FEATURES)}")
    print(f"Target: {TARGET}")
    print(f"Target classes: {', '.join(TARGET_CLASSES)}")
    
    print("\n📊 Features por categoría:")
    
    print("\n  Meteorológicas básicas:")
    print("    - temperatura, humedad, presion, viento, visibilidad, precipitacion")
    
    print("\n  Viento avanzado:")
    print("    - direccion_viento, runway_heading, viento_cruzado, viento_frente, rafagas")
    
    print("\n  Nubes y visibilidad:")
    print("    - techo_nubes, tipo_nubes")
    
    print("\n  Pista y aeropuerto:")
    print("    - estado_pista, altitud_aeropuerto, altitud_densidad")
    
    print("\n  Fenómenos peligrosos:")
    print("    - turbulencia, riesgo_hielo, tormenta_electrica, cizalladura_viento")
    
    print("\n  Temporales:")
    print("    - hora, mes, dia_año, es_noche")
    
    print("\n  Features derivadas:")
    print("    - diferencia_rafagas, spread_temp_dewpoint, ratio_crosswind")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_schema_summary()