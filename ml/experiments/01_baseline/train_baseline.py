"""
Experimento 1: Baseline Models
Entrenamiento de modelos baseline para comparación
"""
import sys
from pathlib import Path

# Agregar root al path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from ml.config.mlflow_config import MLFLOW_TRACKING_URI, EXPERIMENTS

# Configurar MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENTS["baseline"])


def plot_confusion_matrix(y_true, y_pred, classes):
    """Genera y retorna figura de confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión - Baseline')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    
    return fig


def train_baseline_rf():
    """Entrena Random Forest baseline."""
    
    with mlflow.start_run(run_name="random_forest_baseline"):
        # TODO: Cargar tus datos reales aquí
        # from backend.data import load_training_data
        # X_train, X_test, y_train, y_test = load_training_data()
        
        # Por ahora, placeholder
        print("⚠️  TODO: Implementar carga de datos")
        print("��� Este es un template - agrega tus datos")
        
        # Ejemplo de estructura
        model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42,
        }
        
        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForest",
            "dataset": "synthetic",  # Cambiar a "skbo_historical"
            **model_params
        })
        
        # Entrenar modelo
        model = RandomForestClassifier(**model_params)
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        
        # Log metrics
        # mlflow.log_metrics({
        #     "accuracy": accuracy_score(y_test, y_pred),
        #     "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        # })
        
        # Log confusion matrix
        # fig = plot_confusion_matrix(y_test, y_pred, classes=["BAJO", "MEDIO", "ALTO"])
        # mlflow.log_figure(fig, "confusion_matrix.png")
        # plt.close()
        
        # Log model
        # mlflow.sklearn.log_model(model, "model")
        
        print("✅ Baseline model tracked en MLflow")


if __name__ == "__main__":
    train_baseline_rf()
