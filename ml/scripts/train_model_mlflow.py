"""
Script de entrenamiento con tracking MLflow completo
Ejemplo de cómo integrar tu entrenamiento actual con MLflow
"""
import sys
from pathlib import Path

# Agregar root al path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from datetime import datetime

# Importar utils de MLflow
from ml.utils.mlflow_utils import MLflowTracker, setup_dagshub


def load_data(data_path: str):
    """Carga y prepara datos."""
    print(f"📂 Cargando datos desde {data_path}")
    
    # TODO: Ajusta según tu estructura de datos
    df = pd.read_csv(data_path)
    
    # Separar features y target
    X = df.drop('riesgo_operacional', axis=1)
    y = df['riesgo_operacional']
    
    return X, y


def preprocess_data(X, y):
    """Preprocesa datos."""
    print("🔄 Preprocesando datos...")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    return (X_train_scaled, X_test_scaled, 
            y_train_encoded, y_test_encoded,
            scaler, label_encoder)


def train_model_with_tracking(
    X_train, X_test, y_train, y_test,
    model_type: str = "xgboost",
    experiment_name: str = "aerosafe-skbo-production",
    run_name: str = None,
    dagshub_enabled: bool = False
):
    """
    Entrena modelo con tracking completo de MLflow.
    
    Args:
        X_train, X_test, y_train, y_test: Datos de entrenamiento
        model_type: "xgboost" o "random_forest"
        experiment_name: Nombre del experimento MLflow
        run_name: Nombre del run (auto-genera si None)
        dagshub_enabled: Si usar DagHub para tracking remoto
    """
    
    # Configurar DagHub si está habilitado
    if dagshub_enabled:
        tracking_uri = setup_dagshub("YOUR_USERNAME", "aerosafe")
    else:
        tracking_uri = "sqlite:///mlflow.db"
    
    # Inicializar tracker
    tracker = MLflowTracker(experiment_name, tracking_uri)
    
    # Generar run name si no se proporciona
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_type}_{timestamp}"
    
    print(f"🚀 Iniciando entrenamiento: {run_name}")
    
    with tracker.start_run(run_name):
        
        # 1. Log información del dataset
        tracker.log_dataset_info(X_train, X_test, y_train, y_test)
        
        # 2. Configurar y entrenar modelo
        if model_type == "xgboost":
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Model type no soportado: {model_type}")
        
        # Log parámetros del modelo
        tracker.log_model_params(
            model, 
            model_type=model_type,
            additional_params={
                "airport": "SKBO",
                "data_version": "v1.0",
                "training_date": datetime.now().isoformat(),
            }
        )
        
        print("⚙️  Entrenando modelo...")
        model.fit(X_train, y_train)
        
        # 3. Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 4. Log métricas
        print("📊 Calculando métricas...")
        classes = ["BAJO", "MEDIO", "ALTO"]
        tracker.log_metrics_comprehensive(
            y_test, y_pred, y_pred_proba, classes=classes
        )
        
        # 5. Log visualizaciones
        print("📈 Generando visualizaciones...")
        tracker.log_confusion_matrix(y_test, y_pred, classes=classes)
        tracker.log_class_metrics_chart(y_test, y_pred, classes=classes)
        
        # Feature importance si el modelo lo soporta
        if hasattr(model, 'feature_importances_'):
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            # TODO: Reemplazar con tus nombres reales de features
            tracker.log_feature_importance(model, feature_names)
        
        # 6. Log modelo y artifacts
        print("💾 Guardando modelo y artifacts...")
        tracker.log_model_with_artifacts(
            model=model,
            model_name="model",
            # scaler=scaler,  # Descomentar si tienes scaler
            # label_encoder=label_encoder,  # Descomentar si tienes encoder
            # feature_names=feature_names,  # Descomentar si tienes nombres
        )
        
        # 7. Log información adicional
        model_info = {
            "model_type": model_type,
            "airport": "SKBO",
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "classes": ", ".join(classes),
            "training_date": datetime.now().isoformat(),
        }
        tracker.log_model_info(model_info)
        
        print("✅ Entrenamiento completado y tracked en MLflow")
        print(f"📊 Ver resultados: mlflow ui")
        
        return model


def main():
    """Función principal."""
    
    # Configuración
    DATA_PATH = "data/dataset/weather_risk_advanced.csv"
    EXPERIMENT_NAME = "aerosafe-skbo-production"
    MODEL_TYPE = "xgboost"  # o "random_forest"
    DAGSHUB_ENABLED = False  # Cambiar a True para usar DagHub
    
    print("=" * 60)
    print("🔬 AEROSAFE - Entrenamiento con MLflow Tracking")
    print("=" * 60)
    print()
    
    # 1. Cargar datos
    X, y = load_data(DATA_PATH)
    print(f"✅ Datos cargados: {X.shape[0]} muestras, {X.shape[1]} features")
    
    # 2. Preprocesar
    (X_train, X_test, y_train, y_test, 
     scaler, label_encoder) = preprocess_data(X, y)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 3. Entrenar con tracking
    model = train_model_with_tracking(
        X_train, X_test, y_train, y_test,
        model_type=MODEL_TYPE,
        experiment_name=EXPERIMENT_NAME,
        dagshub_enabled=DAGSHUB_ENABLED
    )
    
    # 4. Guardar modelo en producción (opcional)
    save_to_production = input("\n💾 ¿Guardar modelo en models/production/? (y/n): ")
    if save_to_production.lower() == 'y':
        prod_dir = Path("models/production")
        prod_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, prod_dir / "model.pkl")
        joblib.dump(scaler, prod_dir / "scaler.pkl")
        joblib.dump(label_encoder, prod_dir / "label_encoder.pkl")
        
        print(f"✅ Modelo guardado en {prod_dir}")
    
    print()
    print("=" * 60)
    print("🎉 ¡Proceso completado!")
    print("=" * 60)
    print()
    print("Próximos pasos:")
    print("1. mlflow ui")
    print("2. Revisar métricas y visualizaciones")
    print("3. Comparar con experimentos anteriores")
    print()


if __name__ == "__main__":
    main()