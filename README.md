# AeroSafe

Pronóstico de niebla y tormenta para operación aeronáutica. API REST que,
a partir del METAR actual de un aeropuerto, predice la **probabilidad
calibrada** de niebla o tormenta en las próximas 3 horas.

Entrenado sobre **~180.000 observaciones METAR reales** de aeropuertos
colombianos (IEM/NOAA, 2005-2026), con validación temporal, calibración
de probabilidades y estudio de generalización entre aeropuertos.

> **Proyecto académico.** Es un prototipo de investigación validado, no un
> sistema operacional certificado. Antes de citar cualquier cifra, leer
> [`backend/ml/MODEL_CARD.md`](backend/ml/MODEL_CARD.md): documenta el
> rendimiento honesto (bate a la persistencia pero con techo modesto y
> tasa de falsas alarmas alta), los límites de generalización y qué
> faltaría para uso real (validación contra desvíos, aval del IDEAM,
> safety case bajo RAC/Anexo 3 OACI).
>
> Contiene además un **clasificador legado** entrenado con datos
> sintéticos, conservado por transparencia: su accuracy del 94 % es
> circular (un árbol de profundidad 3 alcanza el 95.9 % de su
> rendimiento) y **no** mide capacidad predictiva real.

---

## Arranque rápido

```bash
cd backend

python -m venv .venv
source .venv/Scripts/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # y rellenar OPENWEATHER_API_KEY

uvicorn main:app --reload
```

- API: http://localhost:8000
- Documentación interactiva: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Con Docker:

```bash
cd backend
docker compose up --build
```

---

## Estructura

`backend/` es la raíz de la aplicación: todos los imports son relativos
a ese directorio y desde ahí se ejecutan `uvicorn`, `pytest` y los
scripts de `ml/`.

```
backend/
├── main.py                 Aplicación FastAPI (lifespan, middleware, health)
├── api/
│   ├── dependencies.py     Auth, rate limiting, validaciones, paginación
│   └── routes/             risk, weather, dashboard
├── core/                   Configuración (pydantic-settings) y logging
├── services/
│   ├── forecast_service.py  Pronóstico calibrado desde el METAR actual
│   ├── metar_taf_service.py  Descarga y parseo de METAR/TAF (NOAA)
│   ├── ml_service_v2.py    Clasificador legado (condición actual)
│   └── weather_service.py  Cliente de OpenWeather
├── features/
│   ├── schema.py           Contrato de las 26 features base
│   ├── forecast_features.py  21 features del pronóstico (train == serve)
│   ├── airports.py         Catálogo + cabecera de pista activa
│   ├── wx_codes.py         Intensidad de precipitación desde el METAR
│   ├── build_features.py   Pipeline del clasificador (29 columnas)
│   ├── defaults.py         Completado de payloads parciales
│   └── adapters/           OpenWeather / METAR -> schema canónico
├── models/
│   ├── forecast/           Modelos de pronóstico calibrados (DVC)
│   └── production/          Clasificador legado (DVC)
├── database/               Conexión, base y repositorios
├── batch/                  Predicción por lotes sobre ficheros
├── ml/
│   ├── MODEL_CARD.md       Rendimiento, límites y hallazgos (leer primero)
│   ├── scripts/            Recolección METAR, pronóstico, verificación
│   └── config/             Configuración de MLflow / DagsHub
├── data/                   METAR, datasets de pronóstico (DVC)
└── tests/                  208 tests
```

---

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/risk/predict` | Predice riesgo a partir de condiciones meteorológicas |
| `POST` | `/api/v1/risk/predict/airport/{icao}` | Riesgo con el clima actual del aeropuerto |
| `GET` | `/api/v1/risk/history` | Historial de predicciones persistidas |
| `GET` | `/api/v1/risk/stats` | Estadísticas agregadas por periodo |
| `GET` | `/api/v1/forecast/{icao}` | **Pronóstico de niebla/tormenta a 3h** (probabilidad calibrada) |
| `GET` | `/api/v1/weather/airport/{icao}/metar` | METAR del aeropuerto |
| `GET` | `/health` | Estado real de modelo y base de datos |

Dos modelos, dos endpoints (ver [MODEL_CARD](backend/ml/MODEL_CARD.md)):

- **`/risk/predict`** clasifica la condición **actual** (modelo legado,
  datos sintéticos — con las limitaciones documentadas).
- **`/forecast/{icao}`** predice el **futuro** a 3 horas con el modelo
  entrenado sobre METAR reales y **calibrado**. Es el recomendado.

### Ejemplo — pronóstico

```bash
curl http://localhost:8000/api/v1/forecast/SKBO
```

```json
{
  "icao": "SKBO",
  "horizonte_horas": 3,
  "objetivo": "niebla o tormenta",
  "probabilidad": 0.34,
  "nivel": "MODERADO",
  "alerta": false,
  "condicion_actual": "niebla",
  "modelo_calibrado": true
}
```

La `probabilidad` está calibrada: 0.34 significa ~34% de ocurrencia real.
Aeropuertos soportados: SKBO, SKRG, SKPS, SKMZ.

### Ejemplo

```bash
curl -X POST http://localhost:8000/api/v1/risk/predict \
  -H "Content-Type: application/json" \
  -d '{"temperatura":15.5,"humedad":95,"viento":45,
       "visibilidad":800,"presion":990,"condicion":"tormenta"}'
```

```json
{
  "riesgo": "ALTO",
  "confianza": 0.9988,
  "probabilidades": {"BAJO": 0.0, "MODERADO": 0.0012, "ALTO": 0.9988},
  "factores_riesgo": ["Viento muy fuerte (45.0 km/h)", "Visibilidad muy reducida (800.0m)"],
  "recomendaciones": ["Implementar restricciones operacionales", "..."],
  "model_status": "ml",
  "imputed_features": ["altitud_aeropuerto", "cizalladura_viento", "..."]
}
```

Dos campos merecen atención:

- **`model_status`** — `"ml"` significa que la predicción viene del modelo
  entrenado. `"mock"` significa que viene de reglas heurísticas de
  respaldo, y la respuesta incluye además un `warning`. **Nunca usar una
  respuesta `mock` para nada que importe.**
- **`imputed_features`** — variables que el cliente no aportó y el sistema
  tuvo que estimar. La API expone 5 campos pero el modelo necesita 26, así
  que esta lista suele ser larga. Cuanto más larga, menos respaldada por
  observaciones está la predicción.

---

## Desarrollo

```bash
cd backend

pip install -r requirements-dev.txt
pytest                                   # 208 tests, cobertura mínima 60%
pytest tests/test_api_integration.py -v  # solo integración
```

### Pipeline de pronóstico (datos reales)

```bash
pip install -r requirements-ml.txt
dvc pull                                             # datos y modelos

# 1. recolectar METAR histórico de un aeropuerto (IEM, gratis, sin key)
python -m ml.scripts.collect_metar_history --icao SKBO --desde 2005
# 2. construir el dataset de pronóstico (features en t, etiqueta en t+3h)
python -m ml.scripts.build_forecast_dataset --icao SKBO --horizonte 3
# 3. entrenar y evaluar contra el baseline de persistencia
python -m ml.scripts.train_forecast --icao SKBO --horizonte 3
# 4. calibrar las probabilidades (Brier -67%, ECE -96%)
python -m ml.scripts.calibrate_forecast --icao SKBO --horizonte 3
# 5. verificar con métricas OMM (POD, FAR, CSI, HSS, BSS)
python -m ml.scripts.verify_forecast --icao SKBO --horizonte 3

# generalización entre aeropuertos
python -m ml.scripts.transfer_matrix --aeropuertos SKBO SKRG SKBQ SKCG
```

Los experimentos quedan en MLflow. El **model card** documenta cada
resultado con su baseline honesto: el pronóstico bate a la persistencia,
la calibración vuelve las probabilidades usables, la generalización
funciona entre aeropuertos análogos con datos, y el techo (~0.32 PR-AUC)
resiste a más datos, features de tendencia y satélite GOES.

### Clasificador legado

```bash
python -m ml.scripts.evaluate_model               # baselines + circularidad
python -m ml.scripts.validate_with_metar --offline
```

`evaluate_model` mide cuánta de la accuracy del clasificador se explica
reconstruyendo las reglas con las que se etiquetó su dataset sintético.

---

## Configuración

Toda la configuración vive en `backend/.env` (plantilla en
`.env.example`). Lo relevante para un despliegue real:

| Variable | Por defecto | Nota |
|---|---|---|
| `DEBUG` | `false` | Con `true`, los errores 500 exponen la traza interna |
| `REQUIRE_API_KEY` | `false` | Activar en producción |
| `VALID_API_KEYS` | vacío | Generar con `secrets.token_urlsafe(32)` |
| `RATE_LIMIT_ENABLED` | `true` | 100 peticiones/min por IP, en memoria del proceso |
| `ALLOWED_ORIGINS` | localhost | No usar `*`: la API envía credenciales |

Al arrancar, la aplicación audita su propia configuración de seguridad y
registra en el log cada punto débil que encuentre.

---

## Resultados del pronóstico

Todo medido sobre un conjunto de test **temporal** (2023-2026, posterior
a todo el entrenamiento — sin fuga de futuro). Detalle en el
[model card](backend/ml/MODEL_CARD.md).

- **Bate al baseline operacional.** Frente a la persistencia ("dentro de
  3 h habrá lo mismo que ahora"), el modelo **duplica la detección**
  (POD 0.40 vs 0.22 en SKBO) con *skill* positivo (Brier Skill Score +0.14).
- **Probabilidades calibradas.** La calibración isotónica baja el Brier
  un 67 % y el ECE un 96 %: un 0.30 significa ~30 % de ocurrencia real.
- **Generaliza entre aeropuertos análogos.** Un modelo de SKBO transfiere
  a SKRG reteniendo el ~87 % del margen (ranking), y a Barranquilla
  (costero, pero con niebla de advección) el 91 %. No transfiere donde el
  régimen o los datos difieren: la transferibilidad la deciden el
  fenómeno y los datos, no la geografía.
- **Techo honesto.** ~0.32 PR-AUC y FAR ~0.71 en SKBO. Tres palancas
  probadas empíricamente y descartadas (más METAR, features de tendencia,
  satélite GOES): el límite es la fuente de información, no la cantidad de
  datos.

## Estado del proyecto

| | |
|---|---|
| API | Operativa; `/forecast/{icao}` verificado con METAR real en vivo |
| Modelo de pronóstico | RandomForest calibrado, SKBO/SKRG/SKPS/SKMZ |
| Datos | ~180k METAR reales por aeropuerto (IEM), versionados con DVC |
| Tests | 208 pasando, cobertura mínima 60 % |
| Docker | `compose config` validado; build sin verificar (daemon caído) |
| Tracking | MLflow (local o DagsHub) |

## Qué faltaría para uso operacional real

El model card lo detalla; en resumen, **no es sobre el modelo**:

1. **Validar contra desenlaces reales** (desvíos, go-arounds), no contra
   el METAR futuro. Requiere datos de la Aerocivil/aerolíneas.
2. **Benchmark contra el TAF oficial** — bloqueado: el TAF histórico
   colombiano no está en archivos abiertos, requeriría acceso al IDEAM.
3. **Reducir la tasa de falsas alarmas** (hoy 50-71 %): es lo que provoca
   fatiga de alertas en un controlador.
4. **Marco regulatorio**: OACI Anexo 3, RAC, safety case bajo el SMS del
   proveedor de servicios de navegación aérea.

---

## Licencia y uso

Proyecto académico. **No usar para decisiones operacionales aeronáuticas.**
La información oficial para operación de vuelo es el METAR/TAF publicado
por la autoridad aeronáutica correspondiente (en Colombia, IDEAM/Aerocivil).
