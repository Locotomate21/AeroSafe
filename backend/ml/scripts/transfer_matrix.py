"""
Matriz de generalizacion entre aeropuertos de una misma familia.

Generaliza evaluate_transfer.py de un par (origen, destino) a NxN. La
pregunta que responde: dentro de una familia climatica (p. ej. andinos de
gran altitud), ¿un modelo entrenado en cualquiera de ellos sirve para los
demas?

Para cada par (entrena en A, evalua en B) reporta el PR-AUC de la
transferencia, y lo pone en contexto con dos referencias del destino B:

  - la tasa base de B (suelo: lo que logra el azar),
  - el modelo local de B (techo: lo mejor alcanzable entrenando alli).

La metrica principal es la RETENCION a nivel de ranking: que fraccion del
margen (local - base) alcanza la transferencia. Se usa PR-AUC y no F1
porque el umbral y la calibracion son especificos del sitio (ya
demostrado), asi que comparar rankings es lo justo.

Uso:
    cd backend
    python -m ml.scripts.transfer_matrix --aeropuertos SKBO SKRG SKIP SKPS SKMZ
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BACKEND_DIR = Path(__file__).resolve().parents[2]
FORECAST_DIR = BACKEND_DIR / "data" / "forecast"

CORTE_TEST = 2023
NO_FEATURES = {"objetivo", "timestamp"}


def cargar(icao: str, horizonte: int):
    ruta = FORECAST_DIR / f"forecast_{icao.lower()}_h{horizonte}.csv"
    if not ruta.exists():
        return None
    df = pd.read_csv(ruta, parse_dates=["timestamp"], low_memory=False)
    features = [c for c in df.columns if c not in NO_FEATURES]
    anio = df.timestamp.dt.year
    return {
        "train": df[anio < CORTE_TEST],
        "test": df[anio >= CORTE_TEST],
        "features": features,
    }


def entrenar(datos):
    m = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    m.fit(datos["train"][datos["features"]].values, datos["train"].objetivo.values)
    return m


def main() -> int:
    parser = argparse.ArgumentParser(description="Matriz de transferencia")
    parser.add_argument("--aeropuertos", nargs="+", required=True)
    parser.add_argument("--horizonte", type=int, default=3)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    icaos = [a.upper() for a in args.aeropuertos]

    print("=" * 72)
    print(f"MATRIZ DE GENERALIZACION  (+{args.horizonte}h)")
    print("=" * 72)

    # Cargar todos y descartar los que no tengan dataset.
    datos = {}
    for icao in icaos:
        d = cargar(icao, args.horizonte)
        if d is None:
            print(f"  AVISO: sin dataset para {icao}, se omite.")
            continue
        datos[icao] = d

    icaos = [i for i in icaos if i in datos]
    if len(icaos) < 2:
        print("\n  Hacen falta al menos 2 aeropuertos con dataset.")
        return 1

    # Verificar que todos comparten el mismo esquema de features.
    ref_feats = datos[icaos[0]]["features"]
    for icao in icaos[1:]:
        if datos[icao]["features"] != ref_feats:
            print(f"  ERROR: {icao} tiene features distintas.")
            return 1

    # Un aeropuerto es "referencia valida" solo si tiene suficientes
    # eventos positivos en test para que su modelo local sea fiable. Con
    # muy pocos, el PR-AUC local es ruido y las retenciones contra el se
    # vuelven absurdas (pueden pasar del 100%: un modelo externo "supera"
    # a un local infraentrenado, lo que no dice nada bueno del externo).
    MIN_POSITIVOS = 800

    # --- Perfil de cada aeropuerto ---
    print("\n  Perfil (test 2023-2026):")
    print(f"    {'ICAO':6s}{'n_test':>9s}{'adversos':>10s}{'tasa':>8s}"
          f"{'PR-AUC loc':>12s}{'referencia':>12s}")
    modelos = {}
    base_rate = {}
    pr_local = {}
    n_positivos = {}
    valido = {}
    for icao in icaos:
        d = datos[icao]
        te = d["test"]
        y = te.objetivo.values
        base_rate[icao] = y.mean()
        n_positivos[icao] = int(y.sum())
        modelos[icao] = entrenar(d)
        p = modelos[icao].predict_proba(te[d["features"]].values)[:, 1]
        pr_local[icao] = average_precision_score(y, p)
        valido[icao] = n_positivos[icao] >= MIN_POSITIVOS
        marca = "valida" if valido[icao] else "DEBIL"
        print(f"    {icao:6s}{len(te):>9,d}{n_positivos[icao]:>10,d}{base_rate[icao]:>8.1%}"
              f"{pr_local[icao]:>12.3f}{marca:>12s}")

    debiles = [i for i in icaos if not valido[i]]
    if debiles:
        print(f"\n    Aviso: {', '.join(debiles)} tienen < {MIN_POSITIVOS} eventos en test;"
              f"\n    su modelo local no es una referencia fiable y se excluyen del\n"
              f"    calculo de retencion (aunque aparecen en la matriz).")

    # --- Matriz PR-AUC (fila entrena, columna evalua) ---
    print("\n" + "-" * 72)
    print("PR-AUC:  fila = entrenado en,  columna = evaluado en")
    print("-" * 72)
    print(f"\n  {'train\\test':>12s}" + "".join(f"{i:>9s}" for i in icaos))

    matriz = {}
    for origen in icaos:
        fila = []
        for destino in icaos:
            d = datos[destino]
            X = d["test"][d["features"]].values
            y = d["test"].objetivo.values
            p = modelos[origen].predict_proba(X)[:, 1]
            pr = average_precision_score(y, p)
            matriz[(origen, destino)] = pr
            fila.append(pr)
        marca = lambda o, de, v: f"{v:.3f}" + ("*" if o == de else " ")
        print(f"  {origen:>12s}" + "".join(f"{marca(origen, icaos[j], fila[j]):>9s}"
                                            for j in range(len(icaos))))
    print("\n  (* = diagonal: modelo local, techo de referencia)")

    # --- Retencion de margen: transferencia vs local, sobre la tasa base ---
    print("\n" + "-" * 72)
    print("RETENCION DE MARGEN (ranking):  (transfer - base) / (local - base)")
    print("-" * 72)
    print("\n  Cuanto del margen alcanzable por un modelo local logra la")
    print("  transferencia desde otro aeropuerto. 100% = tan bueno como local.\n")

    # Solo se mide retencion hacia destinos con referencia local valida.
    retenciones_cross = []
    for destino in icaos:
        if not valido[destino]:
            continue
        margen = pr_local[destino] - base_rate[destino]
        if margen <= 0:
            continue
        for origen in icaos:
            if origen == destino:
                continue
            ret = (matriz[(origen, destino)] - base_rate[destino]) / margen
            retenciones_cross.append((origen, destino, ret))
            print(f"  {origen} -> {destino}:  {ret:>5.0%}")

    media = 0.0
    mejor_par = (None, None, -1.0)
    if retenciones_cross:
        valores = [r for _, _, r in retenciones_cross]
        media = float(np.mean(valores))
        mejor_par = max(retenciones_cross, key=lambda x: x[2])
        print(f"\n  Retencion media (destinos con referencia valida): {media:.0%}")
        print(f"  Mejor par: {mejor_par[0]} -> {mejor_par[1]} ({mejor_par[2]:.0%})")

        # Pares fuertes (>=70%) y debiles (<40%), calculados de los datos.
        fuertes = [(o, d, r) for o, d, r in retenciones_cross if r >= 0.70]
        debiles_pares = [(o, d, r) for o, d, r in retenciones_cross if r < 0.40]

        print()
        print("  LECTURA")
        print()
        if fuertes:
            print("  Transfieren BIEN (>=70% del margen local):")
            for o, d, r in sorted(fuertes, key=lambda x: -x[2]):
                print(f"    {o} -> {d}: {r:.0%}  (destino tasa {base_rate[d]:.1%})")
        if debiles_pares:
            print("  Transfieren MAL (<40%):")
            for o, d, r in sorted(debiles_pares, key=lambda x: x[2]):
                print(f"    {o} -> {d}: {r:.0%}  (destino tasa {base_rate[d]:.1%})")
        if debiles:
            print(f"  Destinos con datos insuficientes (excluidos): {', '.join(debiles)}")

        print()
        print("  La transferencia NO la decide la geografia sino dos factores")
        print("  medibles: que ambos aeropuertos tengan el mismo fenomeno con")
        print("  frecuencia parecida, y que haya datos suficientes. Aeropuertos")
        print("  con tasa base similar y datos ricos transfieren; los que carecen")
        print("  del fenomeno objetivo (tasa muy baja) no son un problema de")
        print("  modelo sino de OBJETIVO: ahi hay que redefinir que se predice.")

    if not args.no_mlflow:
        try:
            import mlflow
            from ml.config.mlflow_config import MLFLOW_TRACKING_URI

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("aerosafe-pronostico")
            with mlflow.start_run(run_name=f"matriz_transfer_{len(icaos)}ap_h{args.horizonte}"):
                mlflow.log_param("aeropuertos", ",".join(icaos))
                for (o, d), v in matriz.items():
                    mlflow.log_metric(f"prauc_{o}_{d}", v)
                if retenciones_cross:
                    mlflow.log_metric("retencion_media_cross", media)
            print(f"\n  Registrado en MLflow: {MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"\n  MLflow no disponible ({e}).")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
