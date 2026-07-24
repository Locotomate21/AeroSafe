"""
Construye el dataset de PRONOSTICO a partir del historico METAR.

Cambio de problema respecto al modelo anterior. Antes se predecia la
condicion ACTUAL a partir de variables de la condicion actual, lo que es
circular (si visibilidad es feature y "hay niebla" es la etiqueta, el
modelo solo reaprende visibilidad<1000 -> niebla) y ademas inutil (con el
METAR delante ya sabes que hay niebla).

Aqui la etiqueta es la condicion DENTRO DE N HORAS. Eso:

  - Rompe la circularidad por construccion: el objetivo es una
    observacion futura, no una funcion de las features.
  - Tiene valor operacional: un despachador a las 06:00 necesita saber
    como estara a las 09:00 para decidir combustible de espera o
    alterno.

Objetivo por defecto: presencia de niebla o tormenta a +3h. A ese
horizonte el baseline de persistencia solo logra F1 0.30, asi que hay
margen medible para que el modelo aporte, y sigue siendo fisicamente
predecible desde una sola observacion.

Puntos delicados que este script trata explicitamente:

  1. La etiqueta se empareja por MARCA DE TIEMPO real (t + N horas), no
     por posicion de fila. El 5% de las observaciones no son horarias
     exactas y hay 889 huecos > 3h; emparejar por indice colaria un
     "futuro" que en realidad esta a 6 o 20 horas.

  2. La particion train/test es TEMPORAL, no aleatoria. Un split
     aleatorio pondria las 09:00 en train y las 08:00 y 10:00 en test:
     el modelo memorizaria dias concretos en vez de aprender a
     pronosticar. Se entrena con el pasado y se evalua con el futuro.

Uso:
    cd backend
    python -m ml.scripts.build_forecast_dataset
    python -m ml.scripts.build_forecast_dataset --horizonte 6 --corte 2022
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.forecast_features import (  # noqa: E402
    FEATURES_BASE,
    FEATURES_DERIVADAS,
    add_forecast_features,
    es_adverso,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
METAR_DIR = BACKEND_DIR / "data" / "metar"
SALIDA_DIR = BACKEND_DIR / "data" / "forecast"


def construir(df: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """
    Empareja cada observacion con la de horizonte horas despues.

    Se hace con un merge por timestamp exacto, no con shift(): shift
    asume filas equiespaciadas y aqui no lo estan.
    """
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Las features derivadas se calculan con la MISMA funcion que usa el
    # servicio de la API (features.forecast_features), para que
    # entrenamiento e inferencia coincidan exactamente.
    df = add_forecast_features(df)
    df["adverso"] = df["adverso_actual"]

    # Tabla del futuro: la etiqueta objetivo indexada por su propio
    # timestamp, que luego se busca en t + horizonte.
    futuro = df[["timestamp", "adverso"]].rename(
        columns={"timestamp": "t_futuro", "adverso": "objetivo"}
    )
    df["t_objetivo"] = df["timestamp"] + pd.Timedelta(hours=horizonte)

    emparejado = df.merge(
        futuro, left_on="t_objetivo", right_on="t_futuro", how="inner"
    )

    columnas = FEATURES_BASE + FEATURES_DERIVADAS + ["objetivo", "timestamp"]
    resultado = emparejado[columnas].dropna(subset=FEATURES_BASE + ["objetivo"])
    return resultado.reset_index(drop=True)


def _entrada_por_defecto(icao: str) -> Path:
    """
    Localiza el historico mas COMPLETO de un aeropuerto.

    Se elige por numero de filas, no por orden alfabetico: si conviven
    'metar_skbo_2005_2026.csv' y un 'metar_skbo_2023_2024.csv' de una
    prueba, el orden alfabetico coge el segundo (2023 > 2005) y trunca el
    dataset a dos anios sin avisar. Ese fallo ocurrio y truncó SKBO.
    """
    candidatos = list(METAR_DIR.glob(f"metar_{icao.lower()}_*.csv"))
    if not candidatos:
        return METAR_DIR / f"metar_{icao.lower()}_2005_2026.csv"
    # El fichero con mas lineas es el de mayor cobertura temporal.
    return max(candidatos, key=lambda p: p.stat().st_size)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset de pronostico")
    parser.add_argument("--icao", default="SKBO", help="Aeropuerto")
    parser.add_argument("--horizonte", type=int, default=3, help="Horas de anticipacion")
    parser.add_argument("--corte", type=int, default=2023,
                        help="Primer anio del conjunto de test (temporal)")
    parser.add_argument("--entrada", type=Path, default=None)
    args = parser.parse_args()

    icao = args.icao.lower()
    entrada = args.entrada or _entrada_por_defecto(icao)

    print("=" * 72)
    print(f"DATASET DE PRONOSTICO {args.icao.upper()} - niebla/tormenta a +{args.horizonte}h")
    print("=" * 72)

    if not entrada.exists():
        print(f"\nERROR: falta {entrada}. Ejecutar collect_metar_history o dvc pull.")
        return 1

    df = pd.read_csv(entrada, parse_dates=["timestamp"], low_memory=False)
    print(f"\n  Historico: {len(df):,} observaciones")

    datos = construir(df, args.horizonte)
    print(f"  Pares (t, t+{args.horizonte}h) validos: {len(datos):,}")
    print(f"  Tasa de la clase adversa: {datos.objetivo.mean():.2%}")

    # --- Split temporal ---
    train = datos[datos.timestamp.dt.year < args.corte]
    test = datos[datos.timestamp.dt.year >= args.corte]

    print(f"\n  Split temporal (corte {args.corte}):")
    print(f"    train: {len(train):>8,d}  ({train.timestamp.dt.year.min()}-{train.timestamp.dt.year.max()})  "
          f"adversos {train.objetivo.mean():.2%}")
    print(f"    test : {len(test):>8,d}  ({test.timestamp.dt.year.min()}-{test.timestamp.dt.year.max()})  "
          f"adversos {test.objetivo.mean():.2%}")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    ruta = SALIDA_DIR / f"forecast_{icao}_h{args.horizonte}.csv"
    datos.to_csv(ruta, index=False, encoding="utf-8")
    print(f"\n  Escrito en {ruta.relative_to(BACKEND_DIR)}")
    print(f"\n  Siguiente: python -m ml.scripts.train_forecast --icao {args.icao.upper()} "
          f"--horizonte {args.horizonte}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
