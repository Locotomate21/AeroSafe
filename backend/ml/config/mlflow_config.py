# Archivo: ml/config/mlflow_config.py

"""
Configuración centralizada para MLflow + DagsHub
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ============================================
# DAGSHUB / MLFLOW CONFIG
# ============================================

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "AeroSafe")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

USE_DAGSHUB = bool(DAGSHUB_USERNAME and DAGSHUB_TOKEN)

if USE_DAGSHUB:
    DAGSHUB_URL = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}"
    MLFLOW_TRACKING_URI = f"{DAGSHUB_URL}.mlflow"

    # 👉 Credenciales que MLflow realmente usa
    MLFLOW_TRACKING_USERNAME = DAGSHUB_USERNAME
    MLFLOW_TRACKING_PASSWORD = DAGSHUB_TOKEN
else:
    DAGSHUB_URL = None
    MLFLOW_TRACKING_URI = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{PROJECT_ROOT}/mlflow.db"
    )
    MLFLOW_TRACKING_USERNAME = None
    MLFLOW_TRACKING_PASSWORD = None


# ============================================
# EXPERIMENTOS
# ============================================
EXPERIMENTS = {
    "production": os.getenv("MLFLOW_EXPERIMENT_NAME", "aerosafe-production"),
    "baseline": "aerosafe-baseline",
    "features": "aerosafe-feature-engineering",
    "skbo": "aerosafe-skbo-specific",
}

# ============================================
# ARTIFACTS
# ============================================
PRODUCTION_ARTIFACTS = [
    "model.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "feature_names.txt",
]

# ============================================
# MÉTRICAS A TRACKEAR
# ============================================
METRICS_TO_TRACK = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_ovr",
]

# 🔧 ACTUALIZADO: 3 clases
CLASSES = ["BAJO", "MODERADO", "ALTO"]

# Métricas por clase
CLASS_METRICS = []
for cls in CLASSES:
    CLASS_METRICS.extend([
        f"precision_{cls}",
        f"recall_{cls}",
        f"f1_{cls}",
    ])

# ============================================
# TAGS ÚTILES
# ============================================
DEFAULT_TAGS = {
    "project": "AeroSafe",
    "type": "risk_prediction",
    "domain": "aviation",
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_mlflow_uri():
    """Retorna URI de MLflow tracking"""
    return MLFLOW_TRACKING_URI


def is_dagshub_configured():
    """Verifica si DagsHub está configurado"""
    return bool(DAGSHUB_USERNAME and DAGSHUB_TOKEN)


def print_config():
    """Imprime configuración actual"""
    print("=" * 70)
    print("MLFLOW CONFIGURATION")
    print("=" * 70)
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"DagsHub configured: {is_dagshub_configured()}")
    if is_dagshub_configured():
        print(f"DagsHub URL: {DAGSHUB_URL}")
        print(f"Username: {DAGSHUB_USERNAME}")
    print(f"Default experiment: {EXPERIMENTS['production']}")
    print(f"Classes: {', '.join(CLASSES)}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()