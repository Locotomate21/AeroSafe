"""
Utilidades para MLflow tracking - AeroSafe
"""
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from pathlib import Path
import json


class MLflowTracker:
    """Wrapper para facilitar tracking de experimentos."""
    
    def __init__(self, experiment_name: str, tracking_uri: str = None):
        """
        Inicializa tracker.
        
        Args:
            experiment_name: Nombre del experimento
            tracking_uri: URI de MLflow (None = local, o DagHub URL)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    def start_run(self, run_name: str):
        """Inicia un nuevo run."""
        return mlflow.start_run(run_name=run_name)
    
    def log_model_params(self, model, model_type: str, additional_params: dict = None):
        """
        Log parámetros del modelo.
        
        Args:
            model: Modelo sklearn/xgboost
            model_type: Tipo de modelo ("RandomForest", "XGBoost", etc)
            additional_params: Parámetros adicionales a loggear
        """
        params = {
            "model_type": model_type,
            **model.get_params(),
        }
        
        if additional_params:
            params.update(additional_params)
        
        mlflow.log_params(params)
    
    def log_dataset_info(self, X_train, X_test, y_train, y_test):
        """Log información del dataset."""
        mlflow.log_params({
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "train_class_dist": str(pd.Series(y_train).value_counts().to_dict()),
            "test_class_dist": str(pd.Series(y_test).value_counts().to_dict()),
        })
    
    def log_metrics_comprehensive(self, y_true, y_pred, y_pred_proba=None, classes=None):
        """
        Log métricas comprehensivas.
        
        Args:
            y_true: Labels reales
            y_pred: Predicciones
            y_pred_proba: Probabilidades (opcional)
            classes: Nombres de clases (opcional)
        """
        # Métricas generales
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC si hay probabilidades
        if y_pred_proba is not None:
            try:
                metrics["roc_auc_ovr"] = roc_auc_score(
                    y_true, y_pred_proba, 
                    average='weighted', 
                    multi_class='ovr'
                )
            except Exception as e:
                print(f"⚠️  No se pudo calcular ROC-AUC: {e}")
        
        mlflow.log_metrics(metrics)
        
        # Métricas por clase
        if classes is None:
            classes = np.unique(y_true)
        
        report = classification_report(y_true, y_pred, 
                                       target_names=[str(c) for c in classes],
                                       output_dict=True,
                                       zero_division=0)
        
        for class_name in classes:
            class_str = str(class_name)
            if class_str in report:
                mlflow.log_metrics({
                    f"{class_str}_precision": report[class_str]['precision'],
                    f"{class_str}_recall": report[class_str]['recall'],
                    f"{class_str}_f1": report[class_str]['f1-score'],
                })
    
    def log_confusion_matrix(self, y_true, y_pred, classes=None, filename="confusion_matrix.png"):
        """Genera y loggea confusion matrix."""
        if classes is None:
            classes = np.unique(y_true)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Cantidad'})
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.ylabel('Clase Real', fontsize=12)
        plt.xlabel('Clase Predicha', fontsize=12)
        plt.tight_layout()
        
        mlflow.log_figure(fig, filename)
        plt.close()
    
    def log_feature_importance(self, model, feature_names, top_n=15, filename="feature_importance.png"):
        """Genera y loggea feature importance."""
        if not hasattr(model, 'feature_importances_'):
            print("⚠️  Modelo no tiene feature_importances_")
            return
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.barh(range(top_n), importance[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importancia', fontsize=12)
        plt.title(f'Top {top_n} Features Más Importantes', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        mlflow.log_figure(fig, filename)
        plt.close()
        
        # También guardar como JSON
        importance_dict = {
            feature_names[i]: float(importance[i]) 
            for i in indices
        }
        mlflow.log_dict(importance_dict, "feature_importance.json")
    
    def log_class_metrics_chart(self, y_true, y_pred, classes=None, filename="class_metrics.png"):
        """Genera gráfico de métricas por clase."""
        if classes is None:
            classes = np.unique(y_true)
        
        report = classification_report(y_true, y_pred,
                                       target_names=[str(c) for c in classes],
                                       output_dict=True,
                                       zero_division=0)
        
        # Crear DataFrame con métricas por clase
        metrics_data = []
        for class_name in classes:
            class_str = str(class_name)
            if class_str in report:
                metrics_data.append({
                    'Clase': class_str,
                    'Precision': report[class_str]['precision'],
                    'Recall': report[class_str]['recall'],
                    'F1-Score': report[class_str]['f1-score'],
                })
        
        df = pd.DataFrame(metrics_data)
        
        # Gráfico de barras agrupadas
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, df['Precision'], width, label='Precision', color='skyblue')
        plt.bar(x, df['Recall'], width, label='Recall', color='lightgreen')
        plt.bar(x + width, df['F1-Score'], width, label='F1-Score', color='salmon')
        
        plt.xlabel('Clase de Riesgo', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Métricas por Clase', fontsize=14, fontweight='bold')
        plt.xticks(x, df['Clase'])
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        mlflow.log_figure(fig, filename)
        plt.close()
    
    def log_model_with_artifacts(self, model, model_name: str, 
                                 scaler=None, label_encoder=None, 
                                 feature_names=None):
        """
        Loggea modelo con todos sus artifacts asociados.
        
        Args:
            model: Modelo entrenado
            model_name: Nombre para el artifact
            scaler: Scaler usado (opcional)
            label_encoder: Label encoder (opcional)
            feature_names: Lista de nombres de features (opcional)
        """
        # Log el modelo principal
        mlflow.sklearn.log_model(model, model_name)
        
        # Log preprocessing artifacts si existen
        if scaler is not None:
            import joblib
            scaler_path = Path("/tmp/scaler.pkl")
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path, "preprocessing")
        
        if label_encoder is not None:
            import joblib
            encoder_path = Path("/tmp/label_encoder.pkl")
            joblib.dump(label_encoder, encoder_path)
            mlflow.log_artifact(encoder_path, "preprocessing")
        
        if feature_names is not None:
            features_path = Path("/tmp/feature_names.txt")
            with open(features_path, 'w') as f:
                f.write('\n'.join(feature_names))
            mlflow.log_artifact(features_path)
    
    def log_model_info(self, info_dict: dict, filename="model_info.txt"):
        """Loggea información adicional del modelo."""
        info_path = Path("/tmp") / filename
        
        with open(info_path, 'w') as f:
            for key, value in info_dict.items():
                f.write(f"{key}: {value}\n")
        
        mlflow.log_artifact(info_path)
    
    def compare_runs(self, run_ids: list, metrics: list = None):
        """
        Compara múltiples runs.
        
        Args:
            run_ids: Lista de IDs de runs a comparar
            metrics: Lista de métricas a comparar (None = todas)
        """
        client = mlflow.tracking.MlflowClient()
        
        comparison_data = []
        for run_id in run_ids:
            run = client.get_run(run_id)
            run_data = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                **run.data.metrics
            }
            comparison_data.append(run_data)
        
        df = pd.DataFrame(comparison_data)
        
        if metrics:
            cols = ['run_id', 'run_name'] + metrics
            df = df[cols]
        
        return df


# Función helper para configurar DagHub
def setup_dagshub(repo_owner: str, repo_name: str):
    """
    Configura tracking con DagHub.
    
    Args:
        repo_owner: Tu usuario de DagHub
        repo_name: Nombre del repositorio
    
    Example:
        setup_dagshub("juanperez", "aerosafe")
    """
    import dagshub
    
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    # Obtener tracking URI
    tracking_uri = dagshub.get_env_tracker().tracking_uri
    
    print(f"✅ DagHub configurado")
    print(f"📊 Tracking URI: {tracking_uri}")
    print(f"🌐 Ver en: https://dagshub.com/{repo_owner}/{repo_name}/experiments")
    
    return tracking_uri


# Ejemplo de uso
if __name__ == "__main__":
    # Demo básico
    tracker = MLflowTracker("demo-experiment")
    
    with tracker.start_run("demo-run"):
        mlflow.log_params({"example": "demo"})
        mlflow.log_metrics({"accuracy": 0.95})
    
    print("✅ Demo tracker funcionando")