# ��� MLflow Experiments - AeroSafe

## Estructura de Experimentos

### 01_baseline
Modelos baseline para comparación:
- Random Forest
- XGBoost básico
- Logistic Regression

**Objetivo:** Establecer baseline de performance

### 02_feature_engineering
Experimentación con features:
- Features aeronáuticos (METAR/TAF)
- Mínimos operacionales RAC
- Features temporales
- Feature selection

**Objetivo:** Mejorar performance con domain knowledge

### 03_skbo_production
Modelo final optimizado para SKBO:
- Hiperparámetros tuneados
- Features aeronáuticos optimizados
- Validación con casos reales

**Objetivo:** Modelo production-ready

---

## Cómo ejecutar

```bash
# Baseline
python ml/experiments/01_baseline/train_baseline.py

# Feature engineering
python ml/experiments/02_feature_engineering/train_features.py

# Production model
python ml/experiments/03_skbo_production/train_skbo.py
```

## Ver resultados

```bash
# UI local
mlflow ui

# O en DagHub
# https://dagshub.com/YOUR_USERNAME/aerosafe
```
