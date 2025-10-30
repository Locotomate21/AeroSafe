# scripts/train_model_v2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import os, time

dataset_path = "data/dataset/weather_risk.csv"
model_path = "data/models/risk_model.pkl"

if os.path.exists(model_path):
    dataset_time = os.path.getmtime(dataset_path)
    model_time = os.path.getmtime(model_path)
    if model_time > dataset_time:
        print("✅ El modelo ya está actualizado. No es necesario reentrenar.")
        exit()


print("=" * 60)
print("🚀 ENTRENAMIENTO DE MODELO MEJORADO")
print("=" * 60)

# --- Cargar dataset ---
data_path = "../data/dataset/weather_risk.csv"
df = pd.read_csv(data_path)
print(f"\n✅ Dataset cargado: {df.shape}")

# --- Feature Engineering Avanzado ---
print("\n🔧 Creando features mejoradas...")

# Features base
df['temp_norm'] = (df['temperatura'] - df['temperatura'].mean()) / df['temperatura'].std()
df['hum_norm'] = (df['humedad'] - df['humedad'].mean()) / df['humedad'].std()
df['viento_norm'] = (df['viento'] - df['viento'].mean()) / df['viento'].std()
df['vis_norm'] = (df['visibilidad'] - df['visibilidad'].mean()) / df['visibilidad'].std()

# Interacciones importantes
df['temp_x_hum'] = df['temperatura'] * df['humedad']
df['viento_x_vis'] = df['viento'] * (10000 - df['visibilidad']) / 10000

# Features de riesgo basadas en umbrales
df['viento_alto'] = (df['viento'] > 35).astype(int)
df['viento_medio'] = ((df['viento'] >= 20) & (df['viento'] <= 35)).astype(int)
df['vis_baja'] = (df['visibilidad'] < 3000).astype(int)
df['vis_media'] = ((df['visibilidad'] >= 3000) & (df['visibilidad'] < 6000)).astype(int)
df['temp_extrema'] = ((df['temperatura'] < 5) | (df['temperatura'] > 35)).astype(int)
df['humedad_extrema'] = (df['humedad'] > 90).astype(int)

# Score de riesgo compuesto
df['risk_score'] = (
    (df['viento'] / 50) * 0.3 +
    ((100 - df['humedad']) / 100) * 0.2 +
    ((10000 - df['visibilidad']) / 10000) * 0.35 +
    (abs(df['temperatura'] - 22) / 40) * 0.15
)

# Seleccionar features
feature_cols = [
    'temperatura', 'humedad', 'viento', 'visibilidad',
    'temp_x_hum', 'viento_x_vis', 'risk_score',
    'viento_alto', 'viento_medio', 'vis_baja', 'vis_media',
    'temp_extrema', 'humedad_extrema'
]

X = df[feature_cols].values
y = df['riesgo'].values

# --- Codificar etiquetas ---
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"📊 Clases: {encoder.classes_}")
print(f"📊 Distribución: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")

# --- Split estratificado ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.25,  # 75-25 split
    random_state=42,
    stratify=y_encoded
)

# --- Escalado ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📦 Train: {X_train.shape}, Test: {X_test.shape}")

# --- Modelo 1: Random Forest Optimizado ---
print("\n🌲 Entrenando Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# Validación cruzada
cv_scores_rf = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  CV Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
print(f"  OOB Score: {rf.oob_score_:.4f}")

# --- Modelo 2: XGBoost ---
print("\n⚡ Entrenando XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)
xgb.fit(X_train_scaled, y_train)

cv_scores_xgb = cross_val_score(xgb, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  CV Accuracy: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")

# --- Modelo 3: Decision Tree (interpretable) ---
print("\n🌳 Entrenando Decision Tree...")
dt = DecisionTreeClassifier(
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)
dt.fit(X_train_scaled, y_train)

cv_scores_dt = cross_val_score(dt, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  CV Accuracy: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")

# --- Ensemble Voting ---
print("\n🗳️  Creando Ensemble Voting...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('dt', dt)
    ],
    voting='soft',
    weights=[2, 2, 1]  # RF y XGB tienen más peso
)
ensemble.fit(X_train_scaled, y_train)

# --- Evaluación de todos los modelos ---
models = {
    'Random Forest': rf,
    'XGBoost': xgb,
    'Decision Tree': dt,
    'Ensemble': ensemble
}

print("\n" + "=" * 60)
print("📊 RESULTADOS EN TEST SET")
print("=" * 60)

best_acc = 0
best_model = None
best_name = None

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# --- Evaluación detallada del mejor ---
print("\n" + "=" * 60)
print(f"🏆 MEJOR MODELO: {best_name}")
print("=" * 60)

y_pred_best = best_model.predict(X_test_scaled)
print(f"\n🎯 Accuracy: {best_acc:.4f}")
print("\n📊 Matriz de Confusión:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Calcular métricas por clase
print("\n📈 Métricas por Clase:")
for i, clase in enumerate(encoder.classes_):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  {clase}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=encoder.classes_))

# --- Feature Importance ---
if hasattr(best_model, 'feature_importances_'):
    print("\n🔍 Top 10 Features más importantes:")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for idx, row in importances.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# --- Guardar modelos ---
os.makedirs("data/models", exist_ok=True)

print("\n💾 Guardando modelos...")
joblib.dump(best_model, "data/models/risk_model.pkl")
joblib.dump(rf, "data/models/risk_model_rf.pkl")
joblib.dump(xgb, "data/models/risk_model_xgb.pkl")
joblib.dump(ensemble, "data/models/risk_model_ensemble.pkl")
joblib.dump(scaler, "data/models/scaler.pkl")
joblib.dump(encoder, "data/models/label_encoder.pkl")

with open("data/models/feature_names.txt", "w") as f:
    f.write(",".join(feature_cols))

with open("data/models/model_info.txt", "w") as f:
    f.write(f"Best Model: {best_name}\n")
    f.write(f"Accuracy: {best_acc:.4f}\n")
    f.write(f"Features: {len(feature_cols)}\n")

print(f"\n✅ Modelos guardados en 'data/models/'")
print(f"📈 Accuracy final: {best_acc:.4f}")

if best_acc < 0.65:
    print("\n⚠️  ADVERTENCIA: La precisión es baja (<65%)")
    print("💡 Recomendaciones:")
    print("   1. Ejecuta 'python scripts/analizar_datos.py' para revisar el dataset")
    print("   2. Revisa las reglas de clasificación en 'generate_dataset.py'")
    print("   3. Considera regenerar el dataset con reglas más claras")