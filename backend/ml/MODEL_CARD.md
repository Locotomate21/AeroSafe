# Model Card — AeroSafe Risk Predictor

Documento de referencia sobre qué es y qué no es este modelo. Escrito
para que nadie —incluido su autor dentro de seis meses— le atribuya
capacidades que no tiene.

**Última actualización:** 23 de julio de 2026
**Versión del modelo:** production (`models/production/model.pkl`)

---

## 1. Descripción

| | |
|---|---|
| Tarea | Clasificación multiclase del riesgo meteorológico para operación aeronáutica |
| Algoritmo | `RandomForestClassifier` (scikit-learn 1.7.2) |
| Clases | `BAJO`, `MODERADO`, `ALTO` |
| Features | 29 (26 base + 3 derivadas) |
| Hiperparámetros | `n_estimators=200`, `max_depth=20`, `min_samples_split=10`, `min_samples_leaf=5`, `class_weight='balanced'` |
| Entrenamiento | 5.000 muestras sintéticas, split 80/20 estratificado |

Artefactos que componen el pipeline (los cuatro son necesarios; con el
modelo solo no se puede inferir):

```
models/production/
├── model.pkl              RandomForest entrenado
├── scaler.pkl             StandardScaler ajustado (21 columnas)
├── label_encoder.pkl      dict de 4 LabelEncoders
└── feature_names.txt      orden congelado de las 29 features (UTF-8)
```

---

## 2. Rendimiento declarado

Medido sobre el conjunto de test del dataset sintético
(`ml/scripts/evaluate_model.py`):

| Métrica | Valor |
|---|---|
| Accuracy | 0.9400 |
| F1 (weighted) | 0.9405 |
| Validación cruzada (5 folds) | 0.9460 ± 0.0068 |
| ALTO clasificado como BAJO | 1/200 (0.50 %) |

Por clase:

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BAJO | 0.99 | 0.94 | 0.96 | 450 |
| MODERADO | 0.88 | 0.96 | 0.92 | 350 |
| ALTO | 0.94 | 0.91 | 0.92 | 200 |

**Estas cifras no deben citarse sin la sección 3.**

---

## 3. Limitación principal: el dataset es circular

El conjunto de entrenamiento es sintético. Lo genera
`ml/scripts/generate_dataset_UNIFIED.py`, que produce condiciones
meteorológicas al azar y después les asigna la etiqueta `riesgo` con
`calculate_risk_aviation()`, **una función determinista de umbrales**.

Es decir: la variable objetivo es una función conocida y cerrada de las
variables de entrada. El modelo no está aprendiendo meteorología
aeronáutica; está reconstruyendo esa función.

Medición de cuánto pesa esto:

| Modelo | Accuracy |
|---|---|
| Baseline — clase mayoritaria | 0.4500 |
| Baseline — aleatorio estratificado | 0.3530 |
| **Árbol de decisión, profundidad 3** | **0.9010** |
| RandomForest de producción | 0.9400 |

Un árbol con tres niveles —incapaz de representar nada más complejo que
un puñado de umbrales— alcanza el **95.9 %** del rendimiento del
RandomForest. La conclusión es directa: la tarea consiste en recuperar
unas reglas, y casi toda la accuracy se explica así.

**Consecuencia práctica:** el 0.94 de accuracy mide fidelidad a la
función generadora, no capacidad predictiva sobre condiciones reales. No
sirve como evidencia de que el sistema anticipe riesgo operacional.

Reproducir con:

```bash
cd backend
python -m ml.scripts.evaluate_model
```

---

## 4. Comportamiento con datos reales

`ml/scripts/validate_with_metar.py` pasa METAR reales por el pipeline
completo y los contrasta con los mínimos operacionales RAC/OACI, que son
un criterio **independiente** de las reglas de etiquetado.

Sobre una muestra de 8 METAR de SKBO, SKRG y SKCL:

- 5/8 coincidencias exactas con los mínimos operacionales.
- 2/8 el modelo fue **más conservador** que la norma (aceptable).
- **1/8 el modelo subestimó el riesgo**, que es el error que importa:

  ```
  METAR SKBO 231100Z VRB02KT 1200 BR BKN003 12/11 Q1027
    modelo = MODERADO      mínimos operacionales = ALTO
  ```

  Visibilidad 1200 m con techo roto a 300 ft. El modelo lo consideró
  riesgo moderado.

### Desajuste de distribución

Comparando las condiciones reales con el rango visto en entrenamiento:

- **5 de 8 observaciones caen fuera del rango de `visibilidad` del
  dataset.** El generador produce como máximo 9.996 m, pero el METAR usa
  el código `9999` para "10 km o más", que es el valor más frecuente en
  condiciones normales. El modelo extrapola en el caso más común de
  todos.
- `presion`: media real 1022.6 hPa frente a 1010.7 del entrenamiento.
  SKBO opera sistemáticamente en el extremo alto de la distribución
  sintética.

Un RandomForest no extrapola: ante un valor fuera de rango replica la
hoja más cercana. Las predicciones sobre visibilidad ≥ 9999 m no están
respaldadas por datos de entrenamiento.

---

## 5. Uso previsto

**Adecuado para:**

- Prototipo académico y demostración de una arquitectura MLOps completa
  (API, tracking de experimentos, versionado de datos, tests).
- Herramienta de apoyo informativo, siempre junto al METAR/TAF oficial.

**No adecuado para:**

- Cualquier decisión operacional real: despegue, aterrizaje, desvío o
  cancelación.
- Sustituir el criterio del despachador o del piloto al mando.
- Publicar cifras de rendimiento sin la sección 3 al lado.

---

## 6. Entradas y su procedencia

La API expone 5 variables observables, pero el modelo necesita 26. Las
21 restantes las completa `features/defaults.py`, y cada respuesta las
declara en el campo `imputed_features`.

Las imputaciones son **condicionales a la condición meteorológica**
reportada (perfiles derivados de la mediana/moda del dataset por
`descripcion`). Esto importa: con un perfil benigno global, niebla con
600 m de visibilidad se clasificaba `BAJO` con 0.44 de confianza, porque
veinte variables imputadas como "día perfecto" ahogaban la única señal
adversa. Con perfiles condicionales, el mismo caso da `ALTO` con 0.71.

Aun así, **una predicción basada en 21 valores estimados vale menos que
una basada en observaciones**. Cuando `imputed_features` es largo, la
respuesta debe leerse con esa reserva.

Fórmulas de derivación (idénticas a las del generador, para que
entrenamiento e inferencia calculen lo mismo):

| Variable | Cálculo |
|---|---|
| `viento_cruzado` | `abs(V · sin(Δθ))` |
| `viento_frente` | `V · cos(Δθ)` |
| `altitud_densidad` | `alt + 120 · (T − (15 − alt/1000 · 2))` |
| `punto_rocio` | Magnus-Tetens (b=17.625, c=243.04) |

> Desviación conocida: el generador produce el punto de rocío con ruido
> aleatorio por tramos de humedad, mientras que en inferencia se usa
> Magnus. Se resuelve regenerando el dataset con la fórmula física.

---

## 7. Qué haría falta para que esto fuera defendible

En orden de impacto:

1. **Datos de desenlace real.** METAR históricos de SKBO junto al
   registro de desvíos, cancelaciones y go-arounds del aeropuerto en las
   mismas franjas. Sin esto no existe una variable objetivo legítima, y
   todo lo demás es cosmético.
2. **Reetiquetado por experto.** En su defecto, que un despachador
   etiquete una muestra de METAR reales. Es caro pero rompe la
   circularidad.
3. **Corregir el rango de visibilidad** del generador para cubrir el
   código 9999.
4. **Regenerar el punto de rocío** con Magnus en lugar de ruido.
5. **Calibración de probabilidades** (`CalibratedClassifierCV`): hoy la
   "confianza" es la fracción de votos del bosque, que no es una
   probabilidad calibrada.
6. **Ampliar a más aeropuertos** con sus pistas y elevaciones reales
   (`features/defaults.py::AIRPORTS`).

---

## 8. Reproducibilidad

```bash
cd backend
pip install -r requirements-ml.txt

dvc pull                                  # datos y modelos versionados
python -m ml.scripts.evaluate_model       # evaluación + baselines
python -m ml.scripts.validate_with_metar --offline
python -m ml.scripts.train_model_mlflow   # reentrenar
```

Los experimentos quedan en MLflow (`sqlite:///mlflow.db` en local, o
DagsHub si se configuran las credenciales en `.env`).
