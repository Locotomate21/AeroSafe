import json

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from features.schema_validation import validate_model_schema


def load_model(model_uri: str):
    """
    Carga el modelo y valida que el schema del modelo
    coincida con el schema local.
    """

    # Cargar modelo
    model = mlflow.sklearn.load_model(model_uri)

    # Obtener run_id
    client = MlflowClient()
    model_info = mlflow.models.get_model_info(model_uri)
    run_id = model_info.run_id

    # Descargar schema.json del modelo
    local_schema_path = client.download_artifacts(
        run_id=run_id,
        path="schema.json"
    )

    with open(local_schema_path, "r") as f:
        schema_artifact = json.load(f)

    # Validación crítica
    validate_model_schema(schema_artifact)

    return model
