# Model Card — AeroSafe

Documento de referencia sobre qué es y qué no es este modelo. Escrito
para que nadie —incluido su autor dentro de seis meses— le atribuya
capacidades que no tiene.

**Última actualización:** 23 de julio de 2026

Conviven **dos modelos** con propósitos distintos:

| | Clasificador (production) | Pronóstico (forecast) |
|---|---|---|
| Fichero | `models/production/model.pkl` | `models/forecast/forecast_h3.pkl` |
| Datos | 5.000 sintéticos | 175.329 METAR reales de SKBO |
| Tarea | Riesgo de la condición **actual** | Niebla/tormenta **dentro de 3h** |
| Etiqueta | Reglas deterministas | Observación futura real |
| Estado | Legado, con circularidad (§3) | Recomendado (§9) |

El clasificador se documenta abajo por transparencia sobre sus límites.
El de pronóstico (§9) es el que resuelve el problema de fondo.

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
python -m ml.scripts.train_model_mlflow   # reentrenar el clasificador
```

Los experimentos quedan en MLflow (`sqlite:///mlflow.db` en local, o
DagsHub si se configuran las credenciales en `.env`).

---

## 9. Modelo de pronóstico (recomendado)

Resuelve las dos limitaciones del clasificador: la circularidad (§3) y la
inutilidad de "predecir" una condición que ya se lee en el METAR.

Se sirve en producción vía `GET /api/v1/forecast/{icao}`: descarga el
METAR actual del aeropuerto, aplica el mismo pipeline de features que el
entrenamiento (`features/forecast_features.py`, compartido para evitar
desajuste train/serve) y devuelve la probabilidad **calibrada**.
Aeropuertos servidos: SKBO, SKRG, SKPS, SKMZ.

### Tarea

Predecir la **presencia de niebla o tormenta en SKBO dentro de 3 horas**,
a partir de la observación actual. Clasificación binaria.

- **Datos:** 175.329 observaciones METAR reales del IEM (2005-2026),
  emparejadas (t → t+3h) por marca de tiempo exacta.
- **Split temporal:** entrena con 2005-2022 (114.407 pares), evalúa con
  2023-2026 (24.509). Nunca aleatorio: eso filtraría el futuro.
- **Clase adversa:** 4.51% (desbalance ~1:20), tratada con
  `class_weight='balanced'` y umbral optimizado en train.

### Rendimiento (conjunto de test 2023-2026)

El rival no es la clase mayoritaria sino la **persistencia** ("dentro de
3h habrá lo mismo que ahora"), que es lo que asume un despachador sin
modelo.

| Métrica | Persistencia | Modelo | |
|---|---|---|---|
| F1 | 0.216 | **0.344** | +0.128 |
| Recall | 0.220 | **0.412** | +0.191 |
| Precision | 0.212 | 0.296 | +0.084 |
| PR-AUC | 0.054 (tasa base) | **0.312** | ×5.8 |

El modelo **duplica el recall** de la persistencia: detecta el 41% de los
episodios adversos futuros frente al 22%. PR-AUC 0.312 sobre una tasa
base de 0.054 es casi 6× mejor que el azar.

### Por qué 3 horas

Barrido de horizontes (F1 modelo vs persistencia):

| Horizonte | Modelo | Persistencia | Margen |
|---|---|---|---|
| 1h | 0.515 | 0.498 | +0.017 |
| **3h** | **0.344** | 0.216 | **+0.128** |
| 6h | 0.302 | 0.123 | +0.180 |

A 1h la persistencia ya es casi óptima: el modelo no aporta. A 3h y 6h
gana con claridad. 3h equilibra rendimiento absoluto, margen sobre el
baseline y utilidad operacional (margen para decidir combustible de
espera o alterno).

### Qué features pesan, y por qué importa

```
visibilidad        0.160     hora_sin           0.070
direccion_viento   0.090     spread_t_td        0.067
humedad            0.075     adverso_actual     0.067
```

Dos lecturas relevantes:

- **`precipitacion` ya no domina.** En el clasificador sintético era la
  feature nº 1 (0.175); aquí no entra en el top 10. Confirma que su
  protagonismo anterior era un artefacto del generador, no señal real.
- **`adverso_actual` (la persistencia) pesa solo 0.067.** El modelo no se
  limita a copiar el presente: aprende del ciclo diario (`hora_sin`), la
  saturación (`spread_t_td`) y la advección (`direccion_viento`).

### Generalización a otro aeropuerto (SKBO → SKRG)

Se probó si el modelo entrenado **solo con SKBO** sirve para **SKRG**
(Rionegro) sin reentrenar. Es la prueba de si aprendió física atmosférica
o memorizó El Dorado. SKRG es un buen primer caso: también andino de gran
altitud (2.120 m), pero con **3× más niebla** (13.7% frente a 4.8%).

Evaluado sobre el test de SKRG 2023-2026 (17.483 pares), comparando tres
cosas:

| | Persistencia | Transfer SKBO→SKRG | Local SKRG |
|---|---|---|---|
| F1 | 0.318 | 0.392 | 0.513 |
| PR-AUC | 0.126 (base) | **0.464** | 0.530 |
| Precision | 0.314 | 0.653 | 0.458 |
| Recall | 0.322 | 0.281 | 0.583 |

**Hay que separar dos preguntas distintas:**

- **¿Ordena bien el riesgo?** Sí. En PR-AUC (independiente del umbral) la
  transferencia retiene el **84%** del margen que logra un modelo local
  sobre la tasa base. El modelo de SKBO ordena los episodios de SKRG casi
  tan bien como uno entrenado allí.
- **¿Está bien calibrado su umbral?** No. Con el umbral de SKBO, la
  transferencia es demasiado conservadora en SKRG (precision 0.65,
  recall 0.28): predice pocos eventos porque SKRG tiene mucha más niebla
  de la que el modelo esperaba. Reajustando **solo el umbral** con datos
  de SKRG —sin reentrenar el modelo— el F1 sube de 0.392 a **0.450**,
  cerrando la mitad de la brecha con el modelo local.

**Conclusión:** el modelo aprendió señal atmosférica genuinamente
transferible, no solo peculiaridades de El Dorado. Adaptarlo a un
aeropuerto nuevo es cuestión de **recalibrar el umbral**, que necesita
pocos datos, no de reentrenar desde cero. La brecha que queda tras
recalibrar (0.450 vs 0.513) es la especificidad local real.

### Generalización a la familia andina ampliada (5 aeropuertos)

El resultado limpio de dos aeropuertos invita a una pregunta más dura:
¿se sostiene sobre todos los aeropuertos andinos de gran altitud? Se
probó con cinco (matriz de transferencia 5×5, PR-AUC):

| Aeropuerto | Elev. | Test | Tasa adversa | PR-AUC local | Referencia |
|---|---|---|---|---|---|
| SKBO (Bogotá) | 2548 m | 24.509 | 5.4% | 0.312 | válida |
| SKRG (Rionegro) | 2120 m | 17.483 | 12.6% | 0.530 | válida |
| SKPS (Pasto) | 1795 m | 14.816 | 11.7% | 0.457 | válida |
| SKMZ (Manizales) | 2095 m | 11.999 | 24.8% | 0.585 | válida |
| SKIP (Ipiales) | 2980 m | 15.554 | 2.2% | 0.097 | **débil** (338 eventos) |

**Retención de margen (ranking) hacia cada destino con referencia
válida:**

- **SKBO ↔ SKRG: ~87%** (los dos grandes, registros continuos de ~175k
  observaciones, niebla de radiación de valle).
- SKPS: retiene ~45% desde los grandes.
- SKMZ: ~2-29%. Niebla **orográfica** persistente (24-43% de las horas) y
  deriva temporal train/test: es otro régimen físico, no el mismo con
  otra tasa base.
- **Retención media entre aeropuertos: 34%.**

**El resultado del 87% NO generaliza a la familia ampliada.** Y esa es la
conclusión honesta e informativa: *"andino de gran altitud" no es una
familia homogénea*. Agrupar por elevación es demasiado grueso. Lo que
determina si la transferencia funciona no es compartir cordillera, sino:

1. **Volumen de datos** en origen y destino (SKIP, con 338 eventos, ni
   siquiera tiene un modelo local fiable).
2. **El mismo tipo de niebla** — radiación (SKBO, SKRG) frente a
   orográfica (SKMZ) son físicas distintas.

Es un resultado más valioso que un "87% en todo": acota **cuándo** se
puede reutilizar un modelo (entre aeropuertos análogos y con datos) y
cuándo hay que entrenar uno propio. Que un método declare sus límites es
más útil que uno que aparenta funcionar siempre.

Reproducir:

```bash
python -m ml.scripts.collect_metar_history --icao SKRG --desde 2005
python -m ml.scripts.build_forecast_dataset --icao SKRG --horizonte 3
python -m ml.scripts.evaluate_transfer --origen SKBO --destino SKRG
# matriz completa
python -m ml.scripts.transfer_matrix --aeropuertos SKBO SKRG SKIP SKPS SKMZ
```

### Contraste con otro régimen climático (Caribe)

La prueba más dura: llevar el modelo andino a la costa, un régimen físico
distinto (convección y calor, no niebla de radiación). Se eligieron los
dos grandes del Caribe con datos ricos —Barranquilla (SKBQ) y Cartagena
(SKCG)— para que un eventual fallo no se confunda con escasez de datos,
como sí pasaría con los aeropuertos pequeños de selva o Pacífico.

El resultado **refuta la hipótesis geográfica** y la reemplaza por una
mejor:

| Par | Retención | Tasa base destino |
|---|---|---|
| SKRG → **SKBQ** (costero) | **91%** | 5.2% |
| SKBO → **SKBQ** (costero) | 74% | 5.2% |
| SKBQ (costero) → SKRG | 82% | 12.6% |
| SKCG → SKBO | 32% | 5.4% |
| cualquiera → SKCG | 2-7% (falla) | 1.8% |

**Barranquilla, siendo costero, transfiere tan bien como el mejor par
andino** (SKRG→SKBQ, 91%). No necesita un modelo propio. La razón: está
en la desembocadura del Magdalena y tiene niebla de advección (5.9%, tasa
base casi igual a Bogotá). Mismo fenómeno objetivo, misma frecuencia →
transfiere.

**Cartagena es donde de verdad se rompe**, pero no por ser costero: por
no tener el fenómeno. Allí la niebla es el 0.41% del tiempo (cielo
despejado el 96.4%). El objetivo "niebla o tormenta" queda en 1.8%,
dominado por tormenta, con solo 456 eventos en test — su propio modelo
local es débil (PR-AUC 0.074). No es un problema de modelo sino de
**objetivo**: en Cartagena habría que predecir convección/tormenta, no
niebla. Y aun así el clima adverso de cualquier tipo es raro (máx. 3.6%),
así que es intrínsecamente un problema más difícil.

**Conclusión — el principio que gobierna la transferibilidad no es la
geografía sino dos factores medibles:**

1. **Mismo fenómeno objetivo a frecuencia parecida.** Dos aeropuertos con
   niebla al ~5% transfieren aunque uno sea andino y otro costero
   (SKBO ↔ SKBQ). Dos del mismo bioma no transfieren si su régimen de
   niebla difiere (SKBO vs SKMZ orográfico).
2. **Datos suficientes** en origen y destino.

Corolario para la pregunta "¿un aeropuerto de selva/costa necesita otro
modelo?": **depende de si tiene el fenómeno objetivo con frecuencia
entrenable, no de su bioma.** Donde el fenómeno no existe, el remedio no
es otro modelo sino redefinir qué se predice — y conseguir datos de
desenlace para un objetivo que sí ocurra.

```bash
python -m ml.scripts.transfer_matrix --aeropuertos SKBO SKRG SKBQ SKCG
```

### Calibración de probabilidades

El modelo base usa `class_weight='balanced'`, que le da buen poder de
**ordenación** (PR-AUC alto) pero **probabilidades sin sentido**: como
entrena tratando las clases al 50/50 cuando la real es ~5%, infla los
scores de la clase minoritaria. Sin calibrar, cuando el bosque vota 0.59
el evento ocurre el 10% de las veces.

Cuánto de descalibrado (test SKBO 2023-2026, antes de calibrar):

| Score predicho | Frecuencia real | Gap |
|---|---|---|
| 0.18 | 0.02 | +0.16 |
| 0.36 | 0.05 | +0.32 |
| 0.59 | 0.10 | +0.49 |

El Brier sin calibrar (0.131) es **peor que predecir la tasa base**
(0.051): las probabilidades son inservibles como tal, solo valen para
rankear. Es también por qué el umbral de decisión tenía que estar en 0.71
en vez de 0.5.

**Solución** (`ml/scripts/calibrate_forecast.py`): calibración isotónica
sobre un tramo temporal reservado (entrena 2005-2020, calibra 2021-2022,
evalúa 2023-2026), **sin reentrenar** el modelo.

| Métrica | Sin calibrar | Calibrado (isotonic) | |
|---|---|---|---|
| Brier | 0.131 | **0.043** | −67% |
| ECE | 0.231 | **0.008** | −96% |
| F1 | 0.346 | 0.340 | ≈ igual |

El F1 casi no cambia porque la calibración es monótona: no reordena, solo
corrige la escala. Tras calibrar, una probabilidad de 0.5 significa ~50%
de ocurrencia real, que es lo que un despachador necesita para decidir.

**Conexión con la transferencia:** la calibración es específica del sitio,
igual que el umbral. El modelo de SKBO evaluado sobre SKRG:

| | Brier | ECE |
|---|---|---|
| Sin calibrar | 0.146 | 0.228 |
| Calibrado en SKBO | 0.102 | 0.079 |
| Calibrado en SKRG | **0.087** | **0.009** |

Calibrar en el aeropuerto de origen ayuda, pero no basta porque las tasas
base difieren (4.3% vs 12.6%). Con **2 años del aeropuerto destino** para
recalibrar —sin reentrenar el modelo— la calibración queda casi perfecta.
Refuerza la conclusión de §9: el modelo aprende física transferible, y
adaptarlo a un sitio nuevo es cuestión de recalibrar, no de reentrenar.

### Limitaciones

- **Cinco aeropuertos, todos andinos.** SKBO, SKRG, SKPS, SKMZ, SKIP. La
  transferencia es fuerte solo entre los dos grandes y ricos en datos
  (SKBO, SKRG); degrada hacia los demás. A aeropuertos de perfil distinto
  (costeros: SKCG, SKBQ) está sin probar y se espera peor aún.
- **Solo dos fenómenos.** Niebla y tormenta, que son los que dominan en
  El Dorado. No cubre riesgo por viento (irrelevante en SKBO: 1 obs
  >25 kt en 2 años) ni engelamiento.

### Reproducir

```bash
cd backend
python -m ml.scripts.build_forecast_dataset --horizonte 3
python -m ml.scripts.train_forecast --horizonte 3
```
