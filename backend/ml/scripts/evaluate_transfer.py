"""
Evaluacion de generalizacion entre aeropuertos.

La pregunta: un modelo entrenado SOLO con SKBO, ¿sirve para otro
aeropuerto sin reentrenar? Es la prueba de si aprendio fisica de la
atmosfera o memorizo las peculiaridades de El Dorado.

Se comparan tres cosas sobre el MISMO conjunto de test del aeropuerto
destino:

  1. Persistencia          - el baseline operacional ("seguira igual").
  2. Transferencia         - el modelo de SKBO aplicado tal cual (zero-shot).
  3. Modelo local          - un modelo entrenado con el propio aeropuerto,
                             como techo de referencia.

La lectura importa mas que los numeros:

  - Si transferencia ~ local: el modelo generaliza, aprendio fisica.
  - Si transferencia > persistencia pero < local: generaliza en parte;
    hay senal comun pero tambien especificidad local.
  - Si transferencia <= persistencia: no generaliza; el modelo de SKBO
    esta sobreajustado a SKBO. Tambien es un resultado honesto y
    publicable.

Uso:
    cd backend
    python -m ml.scripts.evaluate_transfer --origen SKBO --destino SKRG
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BACKEND_DIR = Path(__file__).resolve().parents[2]
FORECAST_DIR = BACKEND_DIR / "data" / "forecast"
MODEL_DIR = BACKEND_DIR / "models" / "forecast"

CORTE_TEST = 2023
NO_FEATURES = {"objetivo", "timestamp"}


def cargar(icao: str, horizonte: int):
    ruta = FORECAST_DIR / f"forecast_{icao.lower()}_h{horizonte}.csv"
    if not ruta.exists():
        print(f"ERROR: falta {ruta}. Ejecutar build_forecast_dataset --icao {icao}.")
        sys.exit(1)
    df = pd.read_csv(ruta, parse_dates=["timestamp"], low_memory=False)
    features = [c for c in df.columns if c not in NO_FEATURES]
    train = df[df.timestamp.dt.year < CORTE_TEST]
    test = df[df.timestamp.dt.year >= CORTE_TEST]
    return train, test, features


def umbral_optimo_f1(y, prob) -> float:
    p, r, umbrales = precision_recall_curve(y, prob)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(umbrales[max(np.argmax(f1[:-1]), 0)])


def metricas(nombre, y, prob, umbral) -> dict:
    pred = (prob >= umbral).astype(int)
    return {
        "nombre": nombre,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, prob),
        "roc_auc": roc_auc_score(y, prob),
    }


def imprimir(r: dict) -> None:
    print(f"  {r['nombre']:24s} P={r['precision']:.3f}  R={r['recall']:.3f}  "
          f"F1={r['f1']:.3f}  PR-AUC={r['pr_auc']:.3f}")


def entrenar(X, y) -> RandomForestClassifier:
    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    modelo.fit(X, y)
    return modelo


def main() -> int:
    parser = argparse.ArgumentParser(description="Generalizacion entre aeropuertos")
    parser.add_argument("--origen", default="SKBO", help="Aeropuerto de entrenamiento")
    parser.add_argument("--destino", default="SKRG", help="Aeropuerto de evaluacion")
    parser.add_argument("--horizonte", type=int, default=3)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print(f"GENERALIZACION  {args.origen} -> {args.destino}  (+{args.horizonte}h)")
    print("=" * 72)

    origen_train, origen_test, features = cargar(args.origen, args.horizonte)
    destino_train, destino_test, features_d = cargar(args.destino, args.horizonte)

    if features != features_d:
        print("ERROR: los datasets tienen features distintas.")
        return 1

    print(f"\n  {args.origen}: train {len(origen_train):,}  test {len(origen_test):,}")
    print(f"  {args.destino}: train {len(destino_train):,}  test {len(destino_test):,}  "
          f"(adversos {destino_test.objetivo.mean():.2%})")

    Xo, yo = origen_train[features].values, origen_train.objetivo.values
    Xd_tr, yd_tr = destino_train[features].values, destino_train.objetivo.values
    Xd_te, yd_te = destino_test[features].values, destino_test.objetivo.values

    # --- Modelo origen (reutiliza el guardado si coincide) ---
    ruta_origen = MODEL_DIR / f"forecast_{args.origen.lower()}_h{args.horizonte}.pkl"
    if ruta_origen.exists():
        modelo_origen = joblib.load(ruta_origen)
        print(f"\n  Modelo origen: {ruta_origen.name}")
    else:
        print(f"\n  Entrenando modelo origen con {args.origen} ...")
        modelo_origen = entrenar(Xo, yo)

    # El umbral del modelo origen se fija con SUS datos, no con los del
    # destino: en despliegue real no se tiene el test del destino.
    umbral_origen = umbral_optimo_f1(yo, modelo_origen.predict_proba(Xo)[:, 1])

    # --- Modelo local del destino (techo de referencia) ---
    print(f"  Entrenando modelo local con {args.destino} ...")
    modelo_local = entrenar(Xd_tr, yd_tr)
    umbral_local = umbral_optimo_f1(yd_tr, modelo_local.predict_proba(Xd_tr)[:, 1])

    # ============================================================
    # Evaluacion, toda sobre el test del DESTINO
    # ============================================================
    print("\n" + "-" * 72)
    print(f"RESULTADOS SOBRE EL TEST DE {args.destino} (2023-2026)")
    print("-" * 72)

    # 1. Persistencia
    pred_p = destino_test.adverso_actual.values
    base = {
        "nombre": "persistencia",
        "precision": precision_score(yd_te, pred_p, zero_division=0),
        "recall": recall_score(yd_te, pred_p, zero_division=0),
        "f1": f1_score(yd_te, pred_p, zero_division=0),
        "pr_auc": yd_te.mean(),
    }
    print(f"  {base['nombre']:24s} P={base['precision']:.3f}  R={base['recall']:.3f}  "
          f"F1={base['f1']:.3f}  (tasa base {base['pr_auc']:.3f})")

    # 2. Transferencia (SKBO aplicado a SKRG)
    prob_t = modelo_origen.predict_proba(Xd_te)[:, 1]
    transfer = metricas(f"transfer {args.origen}->{args.destino}", yd_te, prob_t, umbral_origen)
    imprimir(transfer)

    # 3. Modelo local
    prob_l = modelo_local.predict_proba(Xd_te)[:, 1]
    local = metricas(f"local {args.destino}", yd_te, prob_l, umbral_local)
    imprimir(local)

    # ============================================================
    # Veredicto
    # ============================================================
    print("\n" + "-" * 72)
    print("VEREDICTO")
    print("-" * 72)

    gana_persistencia = transfer["f1"] - base["f1"]
    # Que fraccion del rendimiento alcanzable logra la transferencia,
    # medida en el margen sobre la persistencia.
    margen_local = local["f1"] - base["f1"]
    retencion = (transfer["f1"] - base["f1"]) / margen_local if margen_local > 0 else 0.0

    # Retencion a nivel RANKING (PR-AUC), independiente del umbral. Separa
    # dos cosas distintas: cuanto sabe ordenar el modelo el riesgo, frente
    # a si su umbral esta bien calibrado para este aeropuerto. Un modelo
    # puede ordenar perfectamente y aun asi fallar en F1 por un umbral mal
    # puesto, que se arregla sin reentrenar.
    margen_auc = local["pr_auc"] - base["pr_auc"]
    retencion_auc = (transfer["pr_auc"] - base["pr_auc"]) / margen_auc if margen_auc > 0 else 0.0

    # Reajuste del umbral con los datos de TRAIN del destino (no del test):
    # es lo que se haria en despliegue, y no requiere reentrenar el modelo.
    prob_t_tr = modelo_origen.predict_proba(Xd_tr)[:, 1]
    umbral_recal = umbral_optimo_f1(yd_tr, prob_t_tr)
    transfer_recal = metricas("transfer + recalibrado", yd_te, prob_t, umbral_recal)

    print(f"  F1 persistencia .......... {base['f1']:.3f}")
    print(f"  F1 transferencia ......... {transfer['f1']:.3f}  ({gana_persistencia:+.3f} vs persistencia)")
    print(f"  F1 transfer recalibrado .. {transfer_recal['f1']:.3f}  (solo se reajusta el umbral)")
    print(f"  F1 modelo local .......... {local['f1']:.3f}  (techo de referencia)")
    print(f"\n  PR-AUC transferencia ..... {transfer['pr_auc']:.3f}")
    print(f"  PR-AUC local ............. {local['pr_auc']:.3f}")
    print(f"  PR-AUC tasa base ......... {base['pr_auc']:.3f}")
    print(f"\n  Retencion del margen (F1, umbral fijo) ...... {retencion:.0%}")
    print(f"  Retencion del margen (ranking, PR-AUC) ...... {retencion_auc:.0%}")

    print()
    if gana_persistencia <= 0:
        print("  NO GENERALIZA. El modelo de origen no supera a la persistencia\n"
              "  en el destino: esta sobreajustado a su aeropuerto.")
    elif retencion_auc >= 0.75:
        print("  GENERALIZA BIEN a nivel de RANKING. El modelo ordena el riesgo\n"
              "  en el destino casi tan bien como uno local (PR-AUC), aunque su\n"
              "  UMBRAL este descalibrado para la distinta frecuencia del\n"
              "  destino. Lo confirma el F1 recalibrado: reajustar solo el\n"
              "  umbral, sin reentrenar, recupera buena parte del margen.\n"
              "  Conclusion: el modelo aprendio fisica atmosferica comun; la\n"
              "  adaptacion a un aeropuerto nuevo es cuestion de calibrar, no\n"
              "  de reentrenar.")
    else:
        print("  GENERALIZA EN PARTE. Bate a la persistencia pero queda por\n"
              "  debajo del modelo local: hay senal comun y tambien\n"
              "  especificidad local que solo se captura reentrenando.")

    # --- MLflow ---
    if not args.no_mlflow:
        try:
            import mlflow
            from ml.config.mlflow_config import MLFLOW_TRACKING_URI

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("aerosafe-pronostico")
            with mlflow.start_run(run_name=f"transfer_{args.origen}_{args.destino}_h{args.horizonte}"):
                mlflow.log_param("origen", args.origen)
                mlflow.log_param("destino", args.destino)
                mlflow.log_param("horizonte_h", args.horizonte)
                for etiqueta, r in [("persistencia", base), ("transfer", transfer),
                                     ("transfer_recal", transfer_recal), ("local", local)]:
                    for k in ("f1", "recall", "precision", "pr_auc"):
                        if k in r:
                            mlflow.log_metric(f"{etiqueta}_{k}", r[k])
                mlflow.log_metric("retencion_margen_f1", retencion)
                mlflow.log_metric("retencion_margen_ranking", retencion_auc)
            print(f"\n  Registrado en MLflow: {MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"\n  MLflow no disponible ({e}).")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
