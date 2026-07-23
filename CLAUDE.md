# CLAUDE.md

Notas para agentes que trabajen en este repositorio.

## Lo primero

`backend/` es la raíz de la aplicación, no un subdirectorio de un
monorepo. Todo se ejecuta desde ahí:

```bash
cd backend
uvicorn main:app --reload
pytest
python -m ml.scripts.evaluate_model
```

Los imports son relativos a `backend/`: `from core.config import settings`,
no `from backend.core.config import ...`. El repositorio tuvo los dos
estilos mezclados en el mismo fichero y la aplicación no arrancaba de
ninguna de las dos formas. Si añades un import con prefijo `backend.`,
lo rompes otra vez.

## Contrato del pipeline de features

Entrenamiento e inferencia **deben** producir las mismas 29 columnas, en
el mismo orden, con las mismas transformaciones. Este proyecto ya falló
por aquí una vez: la API devolvía 200 con un riesgo plausible mientras
la predicción venía de reglas `if/else`, porque `build_features` no
recibía el scaler y creaba uno nuevo sin ajustar.

Reglas que no se tocan:

- `build_features(df, fit=False)` **exige** `scaler` y `encoders`
  ajustados. Si faltan, lanza `FeaturePipelineError`. No los fabriques.
- `FEATURE_ORDER` está congelado y coincide con
  `models/production/feature_names.txt`. Reordenarlo exige reentrenar.
- Las fórmulas de `features/defaults.py` (viento cruzado, viento de
  frente, altitud de densidad) replican las de
  `ml/scripts/generate_dataset_UNIFIED.py`. Si cambias una, cambia la
  otra o el modelo recibe una distribución que nunca vio.
- Un payload parcial pasa por `complete_raw_features()` **antes** de
  `build_features()`.

## Honestidad en las respuestas

El sistema informa riesgo meteorológico para aviación. Nada puede
presentar una estimación como si fuera una observación:

- Toda predicción lleva `model_status`: `"ml"` o `"mock"`.
- El fallback a mock se registra como `ERROR`, devuelve `confianza=0.0` y
  añade un `warning`. No lo conviertas en warning ni le pongas una
  confianza inventada.
- `imputed_features` declara qué variables se estimaron.
- Las probabilidades salen de `predict_proba()`. Hubo una versión que las
  fabricaba con constantes escritas a mano; no vuelvas ahí.
- `/health` verifica que se pueda inferir de verdad y hace `SELECT 1`
  contra la base. Comprobar que existe el fichero `.pkl` no basta.

## El modelo y sus cifras

Antes de escribir cualquier número de rendimiento en un README, informe
o commit, lee `backend/ml/MODEL_CARD.md`.

Resumen: la accuracy es 0.94, pero un árbol de decisión de profundidad 3
alcanza 0.90 sobre el mismo dataset. Las etiquetas las genera una función
determinista de umbrales, así que el modelo reconstruye esa función en
vez de aprender meteorología. **No cites el 0.94 como evidencia de
capacidad predictiva.**

Para verificarlo: `python -m ml.scripts.evaluate_model`.

## Vocabulario de columnas

Existe uno solo, el de `features/schema.py`: `rafagas`, `techo_nubes`,
`riesgo_hielo`, `viento`, `visibilidad`... En algún momento convivieron
tres (el schema, el del batch con `rafaga`/`nubes`/`hielo`, y el del
adaptador en inglés). Si necesitas traducir desde una API externa, hazlo
en `features/adapters/`.

Unidades: viento en **km/h**, visibilidad en **metros**, presión en
**hPa** (QNH ajustado a nivel del mar), techo en **pies**. OpenWeather
entrega el viento en m/s y el METAR en nudos; ambos se convierten en su
adaptador.

## Seguridad

- `DEBUG` es `false` por defecto y debe seguir siéndolo.
- No pongas API keys de ejemplo en el código. `VALID_API_KEYS` viene
  vacío a propósito.
- `.env` nunca se versiona. `.env.example` sí, sin valores reales.
- `settings.validate_security()` se ejecuta al arrancar; si añades una
  opción de seguridad, añádele su comprobación.

## Tests

```bash
cd backend
pytest                 # 123 tests, umbral de cobertura 60%
```

Al tocar el pipeline de features o el servicio ML, comprueba que sigue
pasando `tests/test_api_integration.py::test_predict_usa_el_modelo_entrenado`.
Ese test existe porque los tests de unidad no detectaron que la API
llevaba tiempo devolviendo predicciones falsas.

El fixture `ml_service` inyecta un modelo simulado pero con **scaler y
encoders reales ajustados**: un mock sin ellos no ejercitaría el
pipeline.

## Datos y modelos

Versionados con DVC, no con git:

```bash
dvc pull      # traer datos y modelos
dvc add ...   # tras regenerar
```

No commitees `.pkl` ni CSV grandes directamente.
