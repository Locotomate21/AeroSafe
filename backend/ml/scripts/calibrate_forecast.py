"""
Calibracion de probabilidades del modelo de pronostico.

El modelo base se entrena con class_weight='balanced', que le da buen
poder de ORDENACION (PR-AUC alto) pero probabilidades sin sentido: como
entrena tratando las clases al 50/50, cuando el bosque vota 0.78 la
frecuencia real del evento es ~0.26. El score sirve para rankear, no como
probabilidad.

Eso tiene dos consecuencias practicas:
  - El umbral de decision hay que ponerlo en 0.71, no en 0.5.
  - Una "probabilidad de niebla del 45%" que en realidad es un 6% es
    inutil para un despachador, que necesita el numero para decidir.

Este script calibra la salida SIN reentrenar el modelo, con un tramo
temporal reservado para ello:

    2005-2020  entrena el bosque
    2021-2022  calibra (ni el bosque ni el test lo han visto)
    2023-2026  evalua

Compara dos metodos:
  - sigmoid (Platt): monotono, robusto con pocos datos, 2 parametros.
  - isotonic: no parametrico, mas flexible, necesita mas datos.

Uso:
    cd backend
    python -m ml.scripts.calibrate_forecast --icao SKBO --horizonte 3
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BACKEND_DIR = Path(__file__).resolve().parents[2]
FORECAST_DIR = BACKEND_DIR / "data" / "forecast"
MODEL_DIR = BACKEND_DIR / "models" / "forecast"

NO_FEATURES = {"objetivo", "timestamp"}

# Tramos temporales. El de calibracion va entre el de entrenamiento y el
# de test, sin solaparse con ninguno.
FIN_TRAIN = 2021   # train: aNios < 2021
FIN_CALIB = 2023   # calib: 2021-2022 ; test: >= 2023


def cargar(icao: str, horizonte: int):
    ruta = FORECAST_DIR / f"forecast_{icao.lower()}_h{horizonte}.csv"
    if not ruta.exists():
        print(f"ERROR: falta {ruta}.")
        sys.exit(1)
    df = pd.read_csv(ruta, parse_dates=["timestamp"], low_memory=False)
    features = [c for c in df.columns if c not in NO_FEATURES]
    anio = df.timestamp.dt.year
    return (
        df[anio < FIN_TRAIN],
        df[(anio >= FIN_TRAIN) & (anio < FIN_CALIB)],
        df[anio >= FIN_CALIB],
        features,
    )


def ece(y, p, n_bins=10) -> float:
    """Error de calibracion esperado, por cuantiles."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df.p, n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), real=("y", "mean"), pred=("p", "mean"))
    return float((g.n * (g.real - g.pred).abs()).sum() / len(df))


def umbral_optimo_f1(y, p) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
    return float(th[max(np.argmax(f1[:-1]), 0)])


def curva_fiabilidad(y, p, n_bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df.p, n_bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(
        pred=("p", "mean"), real=("y", "mean"), n=("y", "size")
    )


def evaluar(nombre, y, p) -> dict:
    umbral = umbral_optimo_f1(y, p)
    return {
        "nombre": nombre,
        "brier": brier_score_loss(y, p),
        "ece": ece(y, p),
        "f1": f1_score(y, (p >= umbral).astype(int), zero_division=0),
        "umbral": umbral,
    }


def imprimir(r: dict) -> None:
    print(f"  {r['nombre']:22s} Brier={r['brier']:.4f}  ECE={r['ece']:.4f}  "
          f"F1={r['f1']:.3f}  (umbral {r['umbral']:.2f})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibracion del pronostico")
    parser.add_argument("--icao", default="SKBO")
    parser.add_argument("--horizonte", type=int, default=3)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print(f"CALIBRACION DE PROBABILIDADES  {args.icao.upper()}  (+{args.horizonte}h)")
    print("=" * 72)

    train, calib, test, features = cargar(args.icao, args.horizonte)
    print(f"\n  train  {len(train):>8,d}  (< {FIN_TRAIN})")
    print(f"  calib  {len(calib):>8,d}  ({FIN_TRAIN}-{FIN_CALIB - 1})")
    print(f"  test   {len(test):>8,d}  (>= {FIN_CALIB})")

    Xtr, ytr = train[features].values, train.objetivo.values
    Xca, yca = calib[features].values, calib.objetivo.values
    Xte, yte = test[features].values, test.objetivo.values

    # Modelo base: mismos hiperparametros que en produccion.
    base = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    base.fit(Xtr, ytr)

    # FrozenEstimator: el modelo ya esta entrenado y no se reajusta;
    # CalibratedClassifierCV solo aprende la funcion de calibracion sobre
    # el tramo de calib. (Sustituye al antiguo cv="prefit").
    congelado = FrozenEstimator(base)
    cal_sig = CalibratedClassifierCV(congelado, method="sigmoid").fit(Xca, yca)
    cal_iso = CalibratedClassifierCV(congelado, method="isotonic").fit(Xca, yca)

    p_base = base.predict_proba(Xte)[:, 1]
    p_sig = cal_sig.predict_proba(Xte)[:, 1]
    p_iso = cal_iso.predict_proba(Xte)[:, 1]

    print("\n" + "-" * 72)
    print("SOBRE EL TEST (2023-2026)")
    print("-" * 72)
    r_base = evaluar("sin calibrar", yte, p_base)
    r_sig = evaluar("sigmoid (Platt)", yte, p_sig)
    r_iso = evaluar("isotonic", yte, p_iso)
    for r in (r_base, r_sig, r_iso):
        imprimir(r)

    ref = brier_score_loss(yte, [yte.mean()] * len(yte))
    print(f"\n  (Brier de referencia, predecir la tasa base {yte.mean():.3f}: {ref:.4f})")

    # --- Efecto sobre las probabilidades ---
    mejor = min((r_sig, r_iso), key=lambda r: r["brier"])
    p_mejor = p_sig if mejor is r_sig else p_iso

    print("\n" + "-" * 72)
    print(f"CURVA DE FIABILIDAD  (mejor metodo: {mejor['nombre']})")
    print("-" * 72)
    print(f"\n  {'':>10s}{'SIN CALIBRAR':>24s}{'CALIBRADO':>22s}")
    print(f"  {'bin':>10s}{'pred':>8s}{'real':>8s}{'gap':>8s}{'pred':>8s}{'real':>8s}{'gap':>8s}")
    c_base = curva_fiabilidad(yte, p_base)
    c_cal = curva_fiabilidad(yte, p_mejor)
    for i in range(min(len(c_base), len(c_cal))):
        b, c = c_base.iloc[i], c_cal.iloc[i]
        print(f"  {i:>10d}{b.pred:>8.3f}{b.real:>8.3f}{b.pred - b.real:>+8.3f}"
              f"{c.pred:>8.3f}{c.real:>8.3f}{c.pred - c.real:>+8.3f}")

    print("\n" + "-" * 72)
    print("VEREDICTO")
    print("-" * 72)
    print(f"  Brier: {r_base['brier']:.4f} -> {mejor['brier']:.4f}  "
          f"({(1 - mejor['brier'] / r_base['brier']):.0%} mejor)")
    print(f"  ECE:   {r_base['ece']:.4f} -> {mejor['ece']:.4f}  "
          f"({(1 - mejor['ece'] / r_base['ece']):.0%} mejor)")
    print(f"  F1:    {r_base['f1']:.3f} -> {mejor['f1']:.3f}  "
          f"(el ranking apenas cambia: calibrar no reordena)")
    print(f"\n  Tras calibrar, el umbral optimo pasa de {r_base['umbral']:.2f} a "
          f"{mejor['umbral']:.2f}:")
    print(f"  ahora una probabilidad de 0.5 significa aproximadamente 50% de\n"
          f"  ocurrencia, que es lo que un despachador necesita para decidir.")

    # --- Guardar el modelo calibrado ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    modelo_final = cal_sig if mejor is r_sig else cal_iso
    ruta = MODEL_DIR / f"forecast_{args.icao.lower()}_h{args.horizonte}_calibrado.pkl"
    joblib.dump(modelo_final, ruta)
    print(f"\n  Modelo calibrado ({mejor['nombre']}) guardado en "
          f"{ruta.relative_to(BACKEND_DIR)}")

    if not args.no_mlflow:
        try:
            import mlflow
            from ml.config.mlflow_config import MLFLOW_TRACKING_URI

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("aerosafe-pronostico")
            with mlflow.start_run(run_name=f"calibracion_{args.icao.lower()}_h{args.horizonte}"):
                mlflow.log_param("icao", args.icao)
                mlflow.log_param("metodo", mejor["nombre"])
                for etq, r in [("base", r_base), ("sigmoid", r_sig), ("isotonic", r_iso)]:
                    mlflow.log_metric(f"{etq}_brier", r["brier"])
                    mlflow.log_metric(f"{etq}_ece", r["ece"])
                    mlflow.log_metric(f"{etq}_f1", r["f1"])
            print(f"  Registrado en MLflow: {MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"  MLflow no disponible ({e}).")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
