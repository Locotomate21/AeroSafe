# Guardar como: ml/scripts/train_model_COMPLETE.py

"""
Script de entrenamiento completo del modelo AeroSafe
Versión unificada con 3 clases: BAJO, MODERADO, ALTO
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

# Agregar path para imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.features.build_features import build_features

# Configuración
DATA_PATH = "data/dataset/weather_risk_aviation.csv"
OUTPUT_DIR = "models/production"
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("=" * 70)
print("🚀 ENTRENAMIENTO MODELO AEROSAFE")
print("=" * 70)

# 1. Cargar datos
print("\n📊 Cargando dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  ✓ {len(df)} muestras cargadas")
print(f"  ✓ {len(df.columns)} variables")

# 2. Separar features y target
print("\n🎯 Preparando datos...")
X_raw = df.drop('riesgo', axis=1)
y = df['riesgo']

print(f"  Distribución de clases:")
for risk_class in ['BAJO', 'MODERADO', 'ALTO']:
    count = (y == risk_class).sum()
    pct = (count / len(y)) * 100
    print(f"    {risk_class}: {count} ({pct:.1f}%)")

# 3. Build features
print("\n🔧 Construyendo features...")
X, artifacts = build_features(X_raw, fit=True)
print(f"  ✓ {X.shape[1]} features creadas")

# 4. Split train/test
print("\n📊 Dividiendo train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  ✓ Train: {len(X_train)} muestras")
print(f"  ✓ Test: {len(X_test)} muestras")

# 5. Entrenar modelo
print("\n🤖 Entrenando Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)

model.fit(X_train, y_train)
print("  ✓ Modelo entrenado")

# 6. Evaluar
print("\n📈 Evaluando modelo...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"  ✓ Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Cross-validation
print("\n🔄 Cross-validation (5-fold)...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"  ✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 8. Feature importance
print("\n🎯 Top 10 Features más importantes:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# 9. Guardar artefactos
print("\n💾 Guardando modelo y artefactos...")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Modelo
joblib.dump(model, f"{OUTPUT_DIR}/model.pkl")
print(f"  ✓ model.pkl")

# Scaler
joblib.dump(artifacts['scaler'], f"{OUTPUT_DIR}/scaler.pkl")
print(f"  ✓ scaler.pkl")

# Encoders
joblib.dump(artifacts['encoders'], f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"  ✓ label_encoder.pkl")

# Feature names
with open(f"{OUTPUT_DIR}/feature_names.txt", 'w') as f:
    f.write('\n'.join(artifacts['feature_names']))
print(f"  ✓ feature_names.txt")

print("\n" + "=" * 70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 70)
print(f"\nModelo guardado en: {OUTPUT_DIR}/")
print(f"Accuracy: {accuracy:.4f}")
print(f"Clases: BAJO, MODERADO, ALTO")
print("\n🎯 Siguiente paso: Probar API con el modelo")
print("   cd backend && python main.py")
print("=" * 70)