import pandas as pd
import mlflow

from features.build_features import build_features
from features.schema import TARGET
from features.schema_logging import log_schema_to_mlflow


# =========================
# Cargar datos
# =========================
raw_df = pd.read_csv("data/dataset/weather_raw.csv")

# =========================
# Construcción de features (contrato)
# =========================
X = build_features(raw_df)
y = raw_df[TARGET]

# =========================
# Entrenamiento + MLflow
# =========================
with mlflow.start_run(run_name="aerosafe_train_v5"):
    # Loggear schema y contrato de features
    log_schema_to_mlflow()

    # -------------------------
    # Aquí va tu entrenamiento
    # -------------------------
    model = train_model(X, y)  # asumo que ya existe

    # -------------------------
    # Log del modelo
    # -------------------------
    mlflow.sklearn.log_model(model, artifact_path="model")
