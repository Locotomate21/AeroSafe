"""
Entrena y evalua el modelo de pronostico de niebla/tormenta.

Todo aqui esta orientado a un problema de eventos raros (la clase adversa
es ~5%), asi que:

  - La metrica principal NO es accuracy. Predecir siempre "no adverso"
    acierta el 95% sin pronosticar nada. Se usa PR-AUC y recall sobre la
    clase adversa.

  - El rival a batir NO es la clase mayoritaria, sino la PERSISTENCIA:
    "dentro de 3h habra lo mismo que ahora". Es lo que hace hoy un
    despachador sin modelo. Un modelo que no supere la persistencia no
    aporta nada.

  - La evaluacion es sobre el conjunto de test TEMPORAL (2023-2026), datos
    posteriores a todo el entrenamiento. Sin fuga de futuro.

Uso:
    cd backend
    python -m ml.scripts.train_forecast --horizonte 3
    python -m ml.scripts.train_forecast --horizonte 3 --no-mlflow
    python -m ml.scripts.train_forecast \
        --config ml/config/experiments/01-forecast-randomforest-skbo-v100-training.yaml

Sin --config se usan los valores por defecto de DEFAULTS, que son los
historicos del script. Con --config, cada campo sale del YAML del
experimento. Precedencia: argumento de linea de comandos > YAML > DEFAULTS.
"""
import argparse
import copy
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
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

# Columnas que no son features.
NO_FEATURES = {"objetivo", "timestamp"}

# Configuracion historica del script, la que se usa cuando no se pasa
# --config. El YAML 01-... la reproduce campo a campo; si tocas algo aqui,
# tocalo alli tambien o los dos caminos dejan de dar el mismo modelo.
DEFAULTS = {
    "icao": "SKBO",
    "horizonte": 3,
    "corte_test": CORTE_TEST,
    "no_features": sorted(NO_FEATURES),
    "seed": 42,
    "model_family": "random_forest",
    "hiperparametros": {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_leaf": 20,
        "class_weight": "balanced",  # compensa el 1:20
        "n_jobs": -1,
        "random_state": 42,
    },
    "use_mlflow": True,
    "experiment_name": "aerosafe-pronostico",
    "run_name": None,  # por defecto: forecast_<icao>_h<horizonte>
}

# Unica familia soportada por este script. Un YAML que pida otra cosa se
# rechaza en vez de entrenar un RandomForest a sus espaldas.
FAMILIA_SOPORTADA = "random_forest"


class ConfigError(ValueError):
    """El YAML de experimento no describe algo que este script pueda entrenar."""


def leer_yaml(ruta: Path) -> dict:
    """Lee el YAML del experimento. PyYAML solo hace falta con --config."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - depende del entorno
        raise ConfigError(
            "--config necesita PyYAML: pip install -r requirements-ml.txt"
        )

    if not ruta.exists():
        raise ConfigError(f"no existe el fichero de configuracion {ruta}")

    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(datos, dict):
        raise ConfigError(f"{ruta} no contiene un mapa YAML")
    return datos


def resolver_config(args) -> dict:
    """
    Funde DEFAULTS, el YAML y los argumentos de linea de comandos.

    Precedencia: CLI > YAML > DEFAULTS. Los flags llegan a None cuando no se
    pasan, que es lo que permite distinguir "no lo dijo" de "lo dijo igual
    que el valor por defecto".
    """
    cfg = copy.deepcopy(DEFAULTS)

    if getattr(args, "config", None):
        y = leer_yaml(Path(args.config))
        datos = y.get("data_source") or {}
        entorno = y.get("environment") or {}
        modelo = y.get("model_config") or {}
        mlops = y.get("mlops_config") or {}

        familia = modelo.get("model_family", cfg["model_family"])
        if familia != FAMILIA_SOPORTADA:
            raise ConfigError(
                f"model_family '{familia}' no soportada por train_forecast; "
                f"este script solo entrena '{FAMILIA_SOPORTADA}'"
            )
        cfg["model_family"] = familia

        for clave in ("icao", "horizonte", "corte_test", "no_features"):
            if datos.get(clave) is not None:
                cfg[clave] = datos[clave]
        if entorno.get("seed") is not None:
            cfg["seed"] = entorno["seed"]

        hiper = modelo.get("hyperparameters")
        if hiper is not None:
            if not isinstance(hiper, dict):
                raise ConfigError("model_config.hyperparameters debe ser un mapa")
            # Sustitucion completa, no fusion: asi el YAML es la descripcion
            # entera del modelo y no hereda por accidente lo que no declara.
            cfg["hiperparametros"] = dict(hiper)

        for clave, destino in (
            ("use_mlflow", "use_mlflow"),
            ("experiment_name", "experiment_name"),
            ("run_name", "run_name"),
        ):
            if mlops.get(clave) is not None:
                cfg[destino] = mlops[clave]

    # La semilla del experimento manda sobre random_state solo si el bloque
    # de hiperparametros no lo declara.
    cfg["hiperparametros"].setdefault("random_state", cfg["seed"])

    # --- CLI: lo ultimo, gana sobre todo lo anterior ---
    if args.icao is not None:
        cfg["icao"] = args.icao
    if args.horizonte is not None:
        cfg["horizonte"] = args.horizonte
    if args.corte_test is not None:
        cfg["corte_test"] = args.corte_test
    if args.no_mlflow:
        cfg["use_mlflow"] = False

    cfg["horizonte"] = int(cfg["horizonte"])
    cfg["corte_test"] = int(cfg["corte_test"])
    cfg["no_features"] = set(cfg["no_features"])
    return cfg


def cargar(horizonte: int, icao: str = "skbo", corte_test: int = CORTE_TEST,
           no_features=NO_FEATURES):
    ruta = FORECAST_DIR / f"forecast_{icao.lower()}_h{horizonte}.csv"
    if not ruta.exists():
        print(f"ERROR: falta {ruta}. Ejecutar build_forecast_dataset primero.")
        sys.exit(1)

    df = pd.read_csv(ruta, parse_dates=["timestamp"], low_memory=False)
    features = [c for c in df.columns if c not in set(no_features)]

    # Corte POR ANO, no aleatorio: es un problema de pronostico y un
    # train_test_split dejaria que el modelo viera el futuro.
    train = df[df.timestamp.dt.year < corte_test]
    test = df[df.timestamp.dt.year >= corte_test]

    return train, test, features


def baseline_persistencia(test: pd.DataFrame) -> dict:
    """
    "Dentro de N horas habra lo mismo que ahora."

    Es el rival honesto: lo que un operador supone por defecto. La
    prediccion es simplemente 'adverso_actual'.
    """
    y = test.objetivo.values
    pred = test.adverso_actual.values
    return {
        "nombre": "persistencia",
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        # La persistencia no da score continuo; su PR-AUC es la tasa base.
        "pr_auc": y.mean(),
    }


def evaluar(nombre, y, prob, umbral) -> dict:
    pred = (prob >= umbral).astype(int)
    return {
        "nombre": nombre,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, prob),
        "roc_auc": roc_auc_score(y, prob),
        "umbral": umbral,
    }


def umbral_optimo_f1(y, prob) -> float:
    """Umbral que maximiza F1 en train, no el 0.5 por defecto."""
    p, r, umbrales = precision_recall_curve(y, prob)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    # precision_recall_curve devuelve un umbral menos que puntos.
    return float(umbrales[max(np.argmax(f1[:-1]), 0)])


def imprimir(res: dict) -> None:
    extra = f"  PR-AUC={res['pr_auc']:.3f}" if "roc_auc" in res else ""
    print(
        f"  {res['nombre']:26s} P={res['precision']:.3f}  R={res['recall']:.3f}  "
        f"F1={res['f1']:.3f}{extra}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena el pronostico")
    # Los defaults viven en DEFAULTS, no aqui: con default=None se distingue
    # "no lo pasaron" de "lo pasaron con el valor por defecto", que es lo que
    # hace posible la precedencia CLI > YAML > DEFAULTS.
    parser.add_argument("--config", default=None,
                        help="YAML de experimento (ml/config/experiments/)")
    parser.add_argument("--icao", default=None)
    parser.add_argument("--horizonte", type=int, default=None)
    parser.add_argument("--corte-test", type=int, default=None,
                        help="primer ano del conjunto de test (corte temporal)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    try:
        cfg = resolver_config(args)
    except ConfigError as e:
        print(f"ERROR de configuracion: {e}")
        return 1

    icao = cfg["icao"]
    horizonte = cfg["horizonte"]

    print("=" * 72)
    print(f"PRONOSTICO DE NIEBLA/TORMENTA A +{horizonte}h - ENTRENAMIENTO")
    print("=" * 72)

    if args.config:
        print("\n  config: " + args.config)

    train, test, features = cargar(horizonte, icao, cfg["corte_test"],
                                   cfg["no_features"])
    print(f"\n  train: {len(train):,} ({train.objetivo.mean():.2%} adversos)  "
          f"test: {len(test):,} ({test.objetivo.mean():.2%} adversos)")
    print(f"  features: {len(features)}")

    X_train, y_train = train[features].values, train.objetivo.values
    X_test, y_test = test[features].values, test.objetivo.values

    # --- Baseline ---
    print("\n" + "-" * 72)
    print("BASELINE (lo que hay que superar)")
    print("-" * 72)
    base = baseline_persistencia(test)
    imprimir(base)

    # --- Modelo ---
    print("\n" + "-" * 72)
    print("MODELO (RandomForest)")
    print("-" * 72)

    try:
        modelo = RandomForestClassifier(**cfg["hiperparametros"])
    except TypeError as e:
        print(f"ERROR de configuracion: hiperparametros invalidos ({e})")
        return 1
    modelo.fit(X_train, y_train)

    prob_train = modelo.predict_proba(X_train)[:, 1]
    prob_test = modelo.predict_proba(X_test)[:, 1]

    # El umbral se fija en TRAIN y se aplica en TEST: elegirlo en test
    # seria hacer trampa.
    umbral = umbral_optimo_f1(y_train, prob_train)
    res = evaluar(f"randomforest (umbral {umbral:.2f})", y_test, prob_test, umbral)
    imprimir(res)

    # Tambien con umbral 0.5, para transparencia.
    res_05 = evaluar("randomforest (umbral 0.50)", y_test, prob_test, 0.5)
    imprimir(res_05)

    # --- Comparacion ---
    print("\n" + "-" * 72)
    print("VEREDICTO")
    print("-" * 72)
    mejora_f1 = res["f1"] - base["f1"]
    mejora_recall = res["recall"] - base["recall"]
    print(f"  F1     modelo {res['f1']:.3f}  vs  persistencia {base['f1']:.3f}   "
          f"({mejora_f1:+.3f})")
    print(f"  Recall modelo {res['recall']:.3f}  vs  persistencia {base['recall']:.3f}   "
          f"({mejora_recall:+.3f})")
    print(f"  PR-AUC modelo {res['pr_auc']:.3f}  vs  tasa base {base['pr_auc']:.3f}")

    if mejora_f1 > 0.03:
        print("\n  El modelo supera a la persistencia de forma clara.")
    elif mejora_f1 > 0:
        print("\n  El modelo supera a la persistencia por margen estrecho.")
    else:
        print("\n  El modelo NO mejora la persistencia. A este horizonte, "
              "suponer\n  continuidad es tan bueno como el modelo.")

    print("\n  Matriz de confusion (umbral optimo):")
    cm = confusion_matrix(y_test, (prob_test >= umbral).astype(int))
    print(f"    {'':16s}{'pred: no':>10s}{'pred: si':>10s}")
    print(f"    {'real: no':16s}{cm[0,0]:>10d}{cm[0,1]:>10d}")
    print(f"    {'real: si':16s}{cm[1,0]:>10d}{cm[1,1]:>10d}")

    print("\n  Top 10 features:")
    imp = (pd.DataFrame({"f": features, "i": modelo.feature_importances_})
           .sort_values("i", ascending=False).head(10))
    for _, row in imp.iterrows():
        print(f"    {row.f:20s} {row.i:.4f}")

    # --- Guardar ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODEL_DIR / f"forecast_{icao.lower()}_h{horizonte}.pkl")
    (MODEL_DIR / f"features_{icao.lower()}_h{horizonte}.txt").write_text(
        "\n".join(features), encoding="utf-8"
    )
    print(f"\n  Modelo guardado en {MODEL_DIR.relative_to(BACKEND_DIR)}/")

    # --- MLflow ---
    if cfg["use_mlflow"]:
        try:
            import mlflow
            from ml.config.mlflow_config import MLFLOW_TRACKING_URI

            run_name = cfg["run_name"] or f"forecast_{icao.lower()}_h{horizonte}"
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(cfg["experiment_name"])
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("icao", icao)
                mlflow.log_param("horizonte_h", horizonte)
                mlflow.log_param("n_train", len(train))
                mlflow.log_param("n_test", len(test))
                mlflow.log_param("split", "temporal")
                mlflow.log_param("corte_test", cfg["corte_test"])
                mlflow.log_param("dataset", "real (IEM METAR SKBO)")
                for k, v in cfg["hiperparametros"].items():
                    mlflow.log_param(f"hp_{k}", v)
                if args.config:
                    mlflow.log_param("config", args.config)
                for k in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
                    mlflow.log_metric(f"modelo_{k}", res[k])
                mlflow.log_metric("baseline_f1", base["f1"])
                mlflow.log_metric("baseline_recall", base["recall"])
                mlflow.log_metric("mejora_f1", mejora_f1)
            print(f"  Registrado en MLflow: {MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"  MLflow no disponible ({e}).")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
