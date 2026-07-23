# AeroSafe

Sistema de predicción de riesgo meteorológico para operación aeronáutica.
API REST sobre un modelo de clasificación entrenado con variables
aeronáuticas (viento cruzado, altitud de densidad, techo de nubes,
cizalladura), pensado para el aeropuerto El Dorado (SKBO).

> **Proyecto académico.** El modelo se entrenó con datos sintéticos y no
> es apto para decisiones operacionales reales. Antes de citar cualquier
> cifra de rendimiento, leer [`backend/ml/MODEL_CARD.md`](backend/ml/MODEL_CARD.md),
> que documenta una limitación importante: la accuracy del 94 % mide
> fidelidad a la función que generó las etiquetas, no capacidad
> predictiva sobre condiciones reales.

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
│   ├── ml_service_v2.py    Carga del modelo e inferencia
│   ├── metar_taf_service.py  Descarga y parseo de METAR/TAF (NOAA)
│   └── weather_service.py  Cliente de OpenWeather
├── features/
│   ├── schema.py           Contrato de las 26 features base
│   ├── build_features.py   Pipeline de features (29 columnas, orden congelado)
│   ├── defaults.py         Completado de payloads parciales
│   └── adapters/           OpenWeather -> schema canónico
├── models/
│   ├── models.py           Tablas SQLAlchemy
│   ├── schemas.py          Schemas Pydantic de la API
│   └── production/         Artefactos del modelo (versionados con DVC)
├── database/               Conexión, base y repositorios
├── batch/                  Predicción por lotes sobre ficheros
├── ml/
│   ├── MODEL_CARD.md       Alcance y limitaciones del modelo
│   ├── scripts/            Generación de dataset, entrenamiento, evaluación
│   └── config/             Configuración de MLflow / DagsHub
├── data/dataset/           Datos (versionados con DVC)
└── tests/                  123 tests
```

---

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/risk/predict` | Predice riesgo a partir de condiciones meteorológicas |
| `POST` | `/api/v1/risk/predict/airport/{icao}` | Riesgo con el clima actual del aeropuerto |
| `GET` | `/api/v1/risk/history` | Historial de predicciones persistidas |
| `GET` | `/api/v1/risk/stats` | Estadísticas agregadas por periodo |
| `GET` | `/api/v1/weather/airport/{icao}/metar` | METAR del aeropuerto |
| `GET` | `/health` | Estado real de modelo y base de datos |

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
pytest                                   # 123 tests, cobertura mínima 60%
pytest tests/test_api_integration.py -v  # solo integración
```

### Entrenamiento y evaluación

```bash
pip install -r requirements-ml.txt

dvc pull                                          # datos y modelos

python -m ml.scripts.evaluate_model               # baselines + circularidad
python -m ml.scripts.validate_with_metar --offline  # contraste con METAR reales
python -m ml.scripts.train_model_mlflow           # reentrenar
```

`evaluate_model` es el que conviene ejecutar antes de citar cualquier
métrica: compara el modelo contra baselines triviales y mide cuánta de
su accuracy se explica simplemente reconstruyendo las reglas con las que
se etiquetó el dataset.

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

## Estado actual

| | |
|---|---|
| API | Operativa, 20 rutas |
| Modelo | RandomForest, 3 clases, 29 features |
| Tests | 123 pasando, 62 % de cobertura |
| Docker | `compose config` validado; build sin verificar |
| Versionado de datos | DVC sobre DagsHub |
| Tracking de experimentos | MLflow (local o DagsHub) |

### Limitaciones conocidas

Documentadas en detalle en [`backend/ml/MODEL_CARD.md`](backend/ml/MODEL_CARD.md):

1. **El dataset es circular.** Las etiquetas las genera una función
   determinista de umbrales; un árbol de decisión de profundidad 3
   alcanza el 95.9 % del rendimiento del RandomForest. La accuracy no
   mide capacidad predictiva real.
2. **Desajuste de distribución.** El generador tope la visibilidad en
   9.996 m, pero el METAR usa el código `9999` para "10 km o más", el
   valor más frecuente en condiciones normales. El modelo extrapola en el
   caso más común.
3. **21 de 26 features se imputan** en una petición típica de la API.
4. **Sin validación contra desenlaces reales** (desvíos, cancelaciones,
   go-arounds).

---

## Licencia y uso

Proyecto académico. No usar para decisiones operacionales aeronáuticas.
La información oficial para operación de vuelo es el METAR/TAF publicado
por la autoridad aeronáutica correspondiente.
