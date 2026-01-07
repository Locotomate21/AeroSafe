"""
Configuración centralizada para MLflow
"""
import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Configuración MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", None)  # "username/aerosafe"

# Experimentos
EXPERIMENTS = {
    "baseline": "aerosafe-baseline",
    "features": "aerosafe-feature-engineering",
    "skbo": "aerosafe-skbo-production",
}

# Artifacts importantes
PRODUCTION_ARTIFACTS = [
    "model.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "feature_names.txt",
]

# Métricas a trackear
METRICS_TO_TRACK = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_ovr",
]

# Métricas por clase
CLASSES = ["BAJO", "MEDIO", "ALTO"]
