# Archivo: ml/scripts/train_model_mlflow.py

"""
Script de entrenamiento con MLflow + DagsHub tracking
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)

# Agregar path para imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from features.build_features import build_features
from ml.config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
    MLFLOW_TRACKING_PASSWORD,
    EXPERIMENTS,
    METRICS_TO_TRACK,
    CLASSES,
    DEFAULT_TAGS,
    is_dagshub_configured,
    print_config
)

BACKEND_DIR = Path(__file__).resolve().parents[2]

# Configuración
DATA_PATH = BACKEND_DIR / "data" / "dataset" / "weather_risk_aviation.csv"
OUTPUT_DIR = BACKEND_DIR / "models" / "production"
    
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Hiperparámetros
PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
}

print("=" * 70)
print("🚀 ENTRENAMIENTO AEROSAFE CON MLFLOW")
print("=" * 70)

# Mostrar configuración
print_config()

# Configurar MLflow
print("\n🔧 Configurando MLflow...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if is_dagshub_configured():
    print("✅ DagsHub configurado - experimentos se guardarán en la nube")
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
else:
    print("⚠️  DagsHub no configurado - usando tracking local")

# Crear o usar experimento
experiment_name = EXPERIMENTS["production"]
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Experimento creado: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"✅ Usando experimento: {experiment_name}")
except Exception as e:
    print(f"⚠️  Error con experimento: {e}")
    print("   Usando experimento por defecto")
    experiment_id = None

# 1. Cargar datos
print("\n📊 Cargando dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  ✓ {len(df)} muestras cargadas")

# 2. Preparar datos
print("\n🎯 Preparando datos...")
X_raw = df.drop('riesgo', axis=1)
y = df['riesgo']

print(f"  Distribución de clases:")
for cls in CLASSES:
    count = (y == cls).sum()
    pct = (count / len(y)) * 100
    print(f"    {cls}: {count} ({pct:.1f}%)")

# 3. Build features
print("\n🔧 Construyendo features...")
X, artifacts = build_features(X_raw, fit=True)
print(f"  ✓ {X.shape[1]} features creadas")

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ============================================
# INICIAR RUN DE MLFLOW
# ============================================

with mlflow.start_run(experiment_id=experiment_id, run_name="aerosafe_rf_v1") as run:
    
    print("\n🤖 Entrenando modelo...")
    print(f"   MLflow Run ID: {run.info.run_id}")
    
    # Log tags
    mlflow.set_tags(DEFAULT_TAGS)
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("dataset_version", "v1.0")
    
    # Log parameters
    mlflow.log_params(PARAMS)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_samples", len(X))
    mlflow.log_param("n_classes", len(CLASSES))
    
    # Entrenar
    model = RandomForestClassifier(**PARAMS)
    model.fit(X_train, y_train)
    print("  ✓ Modelo entrenado")
    
    # Evaluar en test
    print("\n📈 Evaluando modelo...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Métricas generales
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    
    # Métricas por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=CLASSES
    )
    
    for i, cls in enumerate(CLASSES):
        mlflow.log_metric(f"precision_{cls}", precision[i])
        mlflow.log_metric(f"recall_{cls}", recall[i])
        mlflow.log_metric(f"f1_{cls}", f1[i])
        mlflow.log_metric(f"support_{cls}", support[i])
    
    # Métricas weighted
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    mlflow.log_metric("precision_weighted", p_w)
    mlflow.log_metric("recall_weighted", r_w)
    mlflow.log_metric("f1_weighted", f1_w)
    
    # Cross-validation
    print("\n🔄 Cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    mlflow.log_metric("cv_mean", cv_scores.mean())
    mlflow.log_metric("cv_std", cv_scores.std())
    print(f"  ✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance
    print("\n🎯 Feature importance...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log top 10 features
    for idx, row in feature_importance.head(10).iterrows():
        mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
    
    # Guardar feature importance como artifact
    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    os.remove("feature_importance.csv")
    
    # Log classification report como artifact
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")
    os.remove("classification_report.txt")
    
    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv("confusion_matrix.csv")
    mlflow.log_artifact("confusion_matrix.csv")
    os.remove("confusion_matrix.csv")
    
    # Log modelo
    print("\n💾 Guardando modelo en MLflow...")
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="aerosafe-risk-predictor"
    )
    
    # Log artifacts adicionales
    print("\n📦 Guardando artifacts...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Guardar localmente también
    joblib.dump(model, f"{OUTPUT_DIR}/model.pkl")
    joblib.dump(artifacts['scaler'], f"{OUTPUT_DIR}/scaler.pkl")
    joblib.dump(artifacts['encoders'], f"{OUTPUT_DIR}/label_encoder.pkl")
    
    with open(f"{OUTPUT_DIR}/feature_names.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(artifacts['feature_names']))
    
    # Log artifacts a MLflow
    mlflow.log_artifact(f"{OUTPUT_DIR}/scaler.pkl")
    mlflow.log_artifact(f"{OUTPUT_DIR}/label_encoder.pkl")
    mlflow.log_artifact(f"{OUTPUT_DIR}/feature_names.txt")
    
    print("  ✓ Modelo y artifacts guardados")
    
    # URL del run
    if is_dagshub_configured():
        from ml.config.mlflow_config import DAGSHUB_URL
        run_url = f"{DAGSHUB_URL}/#/experiments/{experiment_id}/runs/{run.info.run_id}"
        print(f"\n🌐 Ver experimento en DagsHub:")
        print(f"   {run_url}")

print("\n" + "=" * 70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 70)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Run ID: {run.info.run_id}")
print(f"Modelo guardado en: {OUTPUT_DIR}/")

if is_dagshub_configured():
    print(f"\n🎯 Ver experimentos en: https://dagshub.com/{MLFLOW_TRACKING_USERNAME}/{EXPERIMENTS['production']}")
else:
    print("\n💡 Tip: Configura DagsHub en .env para tracking en la nube")

print("=" * 70)