"""
Evaluacion honesta del modelo de riesgo.

Una accuracy alta, sola, no dice nada. Este script la pone en contexto:

1. Contra baselines triviales. Si predecir siempre la clase mayoritaria
   acierta el 45 %, un modelo con 90 % ha aportado menos de lo que parece.

2. Contra la circularidad del dataset. Las etiquetas de
   weather_risk_aviation.csv las genera calculate_risk_aviation(), una
   funcion determinista de reglas. Un modelo entrenado sobre eso no
   aprende meteorologia: aprende las reglas. Se mide cuanta de la
   accuracy se explica asi entrenando un arbol de profundidad 3, que no
   tiene capacidad para nada mas que capturar reglas.

3. Con validacion cruzada, para separar el rendimiento real del ruido de
   un unico split.

Uso:
    cd backend
    python -m ml.scripts.evaluate_model
    python -m ml.scripts.evaluate_model --no-mlflow
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# Permite ejecutar el script directamente, no solo con -m
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.build_features import build_features  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BACKEND_DIR / "data" / "dataset" / "weather_risk_aviation.csv"
CLASES = ["BAJO", "MODERADO", "ALTO"]
RANDOM_STATE = 42


def titulo(texto: str) -> None:
    print(f"\n{'=' * 72}\n{texto}\n{'=' * 72}")


def evaluar_baselines(X_train, X_test, y_train, y_test) -> dict:
    """
    Baselines triviales. Son el suelo: cualquier modelo tiene que
    superarlos con claridad para justificar su existencia.
    """
    titulo("1. BASELINES TRIVIALES")

    resultados = {}
    baselines = {
        "clase_mayoritaria": DummyClassifier(strategy="most_frequent"),
        "aleatorio_estratificado": DummyClassifier(
            strategy="stratified", random_state=RANDOM_STATE
        ),
        "uniforme": DummyClassifier(strategy="uniform", random_state=RANDOM_STATE),
    }

    for nombre, modelo in baselines.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        resultados[nombre] = {"accuracy": acc, "f1_weighted": f1}
        print(f"  {nombre:26s} accuracy={acc:.4f}  f1={f1:.4f}")

    return resultados


def evaluar_modelo(X_train, X_test, y_train, y_test, X, y) -> dict:
    """Entrena el RandomForest de produccion y lo evalua."""
    titulo("2. MODELO DE PRODUCCION (RandomForest)")

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

    print(f"  accuracy = {acc:.4f}")
    print(f"  f1 (weighted) = {f1:.4f}")

    print("\n  Reporte por clase:")
    print("   ", classification_report(y_test, pred, zero_division=0).replace("\n", "\n    "))

    print("  Matriz de confusion (filas=real, columnas=predicho):")
    cm = confusion_matrix(y_test, pred, labels=CLASES)
    print(f"    {'':12s}" + "".join(f"{c:>12s}" for c in CLASES))
    for i, clase in enumerate(CLASES):
        print(f"    {clase:12s}" + "".join(f"{v:>12d}" for v in cm[i]))

    # El error que importa: llamar BAJO a algo que era ALTO. Un falso
    # negativo en seguridad aeronautica no es simetrico a un falso positivo.
    idx_alto, idx_bajo = CLASES.index("ALTO"), CLASES.index("BAJO")
    alto_como_bajo = int(cm[idx_alto][idx_bajo])
    total_alto = int(cm[idx_alto].sum())
    tasa = alto_como_bajo / total_alto if total_alto else 0.0

    print(
        f"\n  ALTO clasificado como BAJO: {alto_como_bajo}/{total_alto} "
        f"({tasa:.2%})  <- el error critico"
    )

    print("\n  Validacion cruzada (5 folds estratificados):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
    print(f"    accuracy = {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"    por fold = {[f'{s:.4f}' for s in scores]}")

    print("\n  Top 10 features por importancia:")
    importancias = (
        pd.DataFrame({"feature": X.columns, "importancia": modelo.feature_importances_})
        .sort_values("importancia", ascending=False)
        .head(10)
    )
    for _, fila in importancias.iterrows():
        print(f"    {fila['feature']:24s} {fila['importancia']:.4f}")

    return {
        "modelo": modelo,
        "accuracy": acc,
        "f1_weighted": f1,
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "alto_como_bajo": alto_como_bajo,
        "tasa_alto_como_bajo": tasa,
        "confusion_matrix": cm,
        "feature_importance": importancias,
    }


def medir_circularidad(X_train, X_test, y_train, y_test) -> dict:
    """
    Cuanta de la senal es simplemente las reglas del generador.

    Un arbol de profundidad 3 no puede modelar meteorologia: solo puede
    partir el espacio en unas pocas regiones. Si aun asi acierta casi
    tanto como el RandomForest, la conclusion es que la tarea consiste en
    recuperar un puñado de umbrales, no en aprender de datos.
    """
    titulo("3. PRUEBA DE CIRCULARIDAD DEL DATASET")

    print(
        "  Las etiquetas de weather_risk_aviation.csv las produce\n"
        "  calculate_risk_aviation(), una funcion determinista de umbrales.\n"
        "  Se comprueba cuanto de la accuracy se explica solo con eso.\n"
    )

    arbol = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    arbol.fit(X_train, y_train)
    acc_arbol = accuracy_score(y_test, arbol.predict(X_test))

    print(f"  Arbol de decision (profundidad 3): accuracy = {acc_arbol:.4f}")

    print("\n  Reglas que aprendio:")
    reglas = export_text(arbol, feature_names=list(X_train.columns), max_depth=3)
    for linea in reglas.split("\n")[:20]:
        print(f"    {linea}")

    return {"accuracy_arbol_d3": acc_arbol}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluacion honesta del modelo")
    parser.add_argument("--no-mlflow", action="store_true", help="No registrar en MLflow")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: no se encuentra el dataset en {args.data}")
        return 1

    titulo("EVALUACION DEL MODELO AEROSAFE")

    df = pd.read_csv(args.data)
    print(f"  Dataset: {args.data.name}  ({len(df)} muestras)")

    y = df["riesgo"]
    X, _ = build_features(df.drop("riesgo", axis=1), fit=True)
    print(f"  Features: {X.shape[1]}")
    print("\n  Distribucion de clases:")
    for clase in CLASES:
        n = int((y == clase).sum())
        print(f"    {clase:10s} {n:5d}  ({n / len(y):.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    baselines = evaluar_baselines(X_train, X_test, y_train, y_test)
    resultado = evaluar_modelo(X_train, X_test, y_train, y_test, X, y)
    circularidad = medir_circularidad(X_train, X_test, y_train, y_test)

    # ---------- Interpretacion ----------
    titulo("4. INTERPRETACION")

    mejor_baseline = max(b["accuracy"] for b in baselines.values())
    ganancia = resultado["accuracy"] - mejor_baseline
    explicado = circularidad["accuracy_arbol_d3"] / resultado["accuracy"]

    print(f"  Mejor baseline trivial ......... {mejor_baseline:.4f}")
    print(f"  Modelo de produccion ........... {resultado['accuracy']:.4f}")
    print(f"  Ganancia sobre el baseline ..... {ganancia:+.4f}")
    print(f"  Arbol de profundidad 3 ......... {circularidad['accuracy_arbol_d3']:.4f}")
    print(f"  Fraccion explicada por reglas .. {explicado:.1%}")

    print("\n  Lectura:")
    if explicado > 0.90:
        print(
            "    Un arbol de 3 niveles alcanza mas del 90 % del rendimiento del\n"
            "    RandomForest. El modelo esta reconstruyendo la funcion de\n"
            "    etiquetado, no aprendiendo de datos meteorologicos. La accuracy\n"
            "    NO es evidencia de capacidad predictiva sobre condiciones reales."
        )
    elif explicado > 0.75:
        print(
            "    Buena parte del rendimiento se explica con reglas simples.\n"
            "    Hay senal adicional, pero conviene validar con datos reales."
        )
    else:
        print("    El modelo captura estructura que las reglas simples no explican.")

    print(
        "\n    Para sostener cualquier afirmacion sobre rendimiento real hace\n"
        "    falta validar contra METAR historicos con desenlaces operacionales\n"
        "    observados. Ver ml/scripts/validate_with_metar.py"
    )

    # ---------- MLflow ----------
    if not args.no_mlflow:
        try:
            import mlflow

            from ml.config.mlflow_config import MLFLOW_TRACKING_URI

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("aerosafe-evaluacion")

            with mlflow.start_run(run_name="evaluacion_honesta"):
                mlflow.log_param("n_muestras", len(df))
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("dataset", args.data.name)
                mlflow.log_param("dataset_tipo", "sintetico")

                mlflow.log_metric("accuracy", resultado["accuracy"])
                mlflow.log_metric("f1_weighted", resultado["f1_weighted"])
                mlflow.log_metric("cv_mean", resultado["cv_mean"])
                mlflow.log_metric("cv_std", resultado["cv_std"])
                mlflow.log_metric("tasa_alto_como_bajo", resultado["tasa_alto_como_bajo"])

                for nombre, metricas in baselines.items():
                    mlflow.log_metric(f"baseline_{nombre}", metricas["accuracy"])

                mlflow.log_metric("accuracy_arbol_d3", circularidad["accuracy_arbol_d3"])
                mlflow.log_metric("ganancia_sobre_baseline", ganancia)
                mlflow.log_metric("fraccion_explicada_por_reglas", explicado)

                mlflow.set_tag(
                    "advertencia",
                    "Dataset sintetico con etiquetas deterministas. La accuracy "
                    "no mide capacidad predictiva sobre condiciones reales.",
                )

            print(f"\n  Registrado en MLflow: {MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"\n  MLflow no disponible ({e}). Usar --no-mlflow para omitirlo.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
