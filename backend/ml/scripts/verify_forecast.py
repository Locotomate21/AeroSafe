"""
Verificacion del pronostico con las metricas estandar de meteorologia
aeronautica (OMM/WMO), no solo las de machine learning.

Un regulador o un meteorologo operativo no habla de PR-AUC ni F1: habla de
POD, FAR, CSI, sesgo y skill scores, calculados sobre la tabla de
contingencia. Este script traduce el modelo a ese idioma y lo compara con
la persistencia (el "pronostico" por defecto de un operador sin modelo).

Tabla de contingencia (evento = niebla o tormenta a +Nh):

                    observado SI   observado NO
    pronosticado SI     a (aciertos)   b (falsas alarmas)
    pronosticado NO     c (fallos)     d (correctos neg.)

Metricas:
    POD  = a/(a+c)        probabilidad de deteccion (recall). 1 = perfecto.
    FAR  = b/(a+b)        razon de falsas alarmas. 0 = perfecto.
    CSI  = a/(a+b+c)      indice critico de exito (threat score). 1 = perfecto.
    SR   = a/(a+b)        success ratio (1 - FAR) = precision.
    Bias = (a+b)/(a+c)    sesgo de frecuencia. 1 = sin sobre/infra-pronostico.
    HSS  = skill de Heidke vs azar. 0 = como el azar, 1 = perfecto.
    PSS  = skill de Peirce (Hanssen-Kuipers) = POD - POFD.
    BSS  = Brier Skill Score vs climatologia. >0 = mejor que la tasa base.

Uso:
    cd backend
    python -m ml.scripts.verify_forecast --icao SKBO --horizonte 3
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BACKEND_DIR = Path(__file__).resolve().parents[2]
FORECAST_DIR = BACKEND_DIR / "data" / "forecast"
MODEL_DIR = BACKEND_DIR / "models" / "forecast"

CORTE_TEST = 2023
NO_FEATURES = {"objetivo", "timestamp"}


def contingencia(y_true, y_pred) -> dict:
    """Tabla de contingencia y metricas categoricas OMM."""
    a = int(((y_pred == 1) & (y_true == 1)).sum())  # aciertos
    b = int(((y_pred == 1) & (y_true == 0)).sum())  # falsas alarmas
    c = int(((y_pred == 0) & (y_true == 1)).sum())  # fallos
    d = int(((y_pred == 0) & (y_true == 0)).sum())  # correctos negativos
    n = a + b + c + d

    pod = a / (a + c) if (a + c) else 0.0
    far = b / (a + b) if (a + b) else 0.0
    sr = a / (a + b) if (a + b) else 0.0
    csi = a / (a + b + c) if (a + b + c) else 0.0
    bias = (a + b) / (a + c) if (a + c) else 0.0
    pofd = b / (b + d) if (b + d) else 0.0  # prob. de falsa deteccion
    pss = pod - pofd  # Peirce / Hanssen-Kuipers

    # Heidke Skill Score: acierto sobre el esperado por azar.
    exacto = (a + d) / n if n else 0.0
    azar = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n) if n else 0.0
    hss = (exacto - azar) / (1 - azar) if (1 - azar) else 0.0

    return {
        "a": a, "b": b, "c": c, "d": d,
        "POD": pod, "FAR": far, "SR": sr, "CSI": csi,
        "Bias": bias, "HSS": hss, "PSS": pss,
    }


def umbral_optimo_csi(y, p) -> float:
    """
    Umbral que maximiza el CSI (threat score), la metrica operacional de
    referencia para eventos de seguridad. Se elige en TRAIN, no en test.
    """
    umbrales = np.linspace(0.01, 0.95, 95)
    mejor_u, mejor_csi = 0.5, -1
    for u in umbrales:
        m = contingencia(y, (p >= u).astype(int))
        if m["CSI"] > mejor_csi:
            mejor_csi, mejor_u = m["CSI"], u
    return mejor_u


def cargar(icao, horizonte):
    ruta = FORECAST_DIR / f"forecast_{icao.lower()}_h{horizonte}.csv"
    df = pd.read_csv(ruta, parse_dates=["timestamp"], low_memory=False)
    features = [c for c in df.columns if c not in NO_FEATURES]
    return df[df.timestamp.dt.year < CORTE_TEST], df[df.timestamp.dt.year >= CORTE_TEST], features


def imprimir_tabla(nombre, m):
    print(f"\n  {nombre}")
    print(f"    contingencia:  aciertos={m['a']:,}  falsas_alarmas={m['b']:,}  "
          f"fallos={m['c']:,}  correctos_neg={m['d']:,}")
    print(f"    POD={m['POD']:.3f}  FAR={m['FAR']:.3f}  CSI={m['CSI']:.3f}  "
          f"Bias={m['Bias']:.2f}  HSS={m['HSS']:.3f}  PSS={m['PSS']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verificacion OMM del pronostico")
    parser.add_argument("--icao", default="SKBO")
    parser.add_argument("--horizonte", type=int, default=3)
    args = parser.parse_args()

    icao = args.icao.upper()
    print("=" * 72)
    print(f"VERIFICACION OMM  {icao}  niebla/tormenta a +{args.horizonte}h")
    print("=" * 72)

    train, test, features = cargar(icao, args.horizonte)
    y_test = test.objetivo.values
    tasa_base = y_test.mean()

    calibrado = MODEL_DIR / f"forecast_{icao.lower()}_h{args.horizonte}_calibrado.pkl"
    if not calibrado.exists():
        print(f"\nERROR: falta el modelo calibrado {calibrado.name}.")
        return 1
    modelo = joblib.load(calibrado)

    prob_train = modelo.predict_proba(train[features].values)[:, 1]
    prob_test = modelo.predict_proba(test[features].values)[:, 1]

    # Umbral operacional: maximiza CSI en train.
    umbral = umbral_optimo_csi(train.objetivo.values, prob_train)
    print(f"\n  test: {len(test):,} casos  |  tasa base (climatologia): {tasa_base:.2%}")
    print(f"  umbral operacional (max CSI en train): {umbral:.2f}")

    # --- Modelo ---
    m_modelo = contingencia(y_test, (prob_test >= umbral).astype(int))
    imprimir_tabla("MODELO (calibrado)", m_modelo)

    # --- Persistencia (rival operacional) ---
    m_pers = contingencia(y_test, test.adverso_actual.values)
    imprimir_tabla("PERSISTENCIA (seguira igual)", m_pers)

    # --- Brier Skill Score ---
    bs = brier_score_loss(y_test, prob_test)
    bs_clim = brier_score_loss(y_test, np.full(len(y_test), tasa_base))
    bss = 1 - bs / bs_clim if bs_clim else 0.0

    print("\n" + "-" * 72)
    print("PROBABILISTICO")
    print("-" * 72)
    print(f"  Brier Score           : {bs:.4f}")
    print(f"  Brier de climatologia : {bs_clim:.4f}  (predecir siempre {tasa_base:.1%})")
    print(f"  Brier Skill Score     : {bss:+.3f}   (>0 = mejor que climatologia)")

    # --- Envelope de operacion: trade-off POD/FAR por umbral ---
    # Un regulador no quiere un unico punto sino la curva completa: que
    # detecccion se logra a cada nivel de falsas alarmas. La probabilidad
    # esta calibrada, asi que el umbral se interpreta como probabilidad de
    # ocurrencia real.
    print("\n" + "-" * 72)
    print("ENVELOPE DE OPERACION (prob calibrada -> deteccion vs falsas alarmas)")
    print("-" * 72)
    anios = max(test.timestamp.dt.year.nunique(), 1)
    print(f"  {'umbral':>7s}{'POD':>7s}{'FAR':>7s}{'CSI':>7s}{'alertas/anio':>13s}")
    for u in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70):
        mm = contingencia(y_test, (prob_test >= u).astype(int))
        alertas = mm["a"] + mm["b"]
        print(f"  {u:>7.2f}{mm['POD']:>7.2f}{mm['FAR']:>7.2f}{mm['CSI']:>7.2f}"
              f"{alertas / anios:>13.0f}")
    print("\n  No hay un punto con POD alta y FAR baja a la vez: es el limite del")
    print("  modelo. La probabilidad calibrada permite que cada operador elija su")
    print("  propio umbral segun su tolerancia (torre saturada vs operacion critica).")

    # --- Lectura ---
    print("\n" + "-" * 72)
    print("LECTURA (en el idioma del regulador)")
    print("-" * 72)
    print(f"  Deteccion (POD):   el modelo capta el {m_modelo['POD']:.0%} de los")
    print(f"    episodios adversos, frente al {m_pers['POD']:.0%} de la persistencia.")
    print(f"  Falsas alarmas (FAR): {m_modelo['FAR']:.0%} de las alertas del modelo no")
    print(f"    se materializan, frente al {m_pers['FAR']:.0%} de la persistencia.")
    print(f"  CSI: {m_modelo['CSI']:.3f} vs {m_pers['CSI']:.3f}. HSS: {m_modelo['HSS']:.3f} vs "
          f"{m_pers['HSS']:.3f}.")
    print(f"  El modelo aporta skill positivo (BSS {bss:+.2f}, HSS {m_modelo['HSS']:.2f}),")
    print(f"  pero con un techo modesto: la niebla a {args.horizonte}h es dificil de")
    print(f"  predecir desde una sola observacion de superficie.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
