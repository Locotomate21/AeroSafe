import mlflow
from features.schema import SCHEMA_VERSION, FEATURES, FEATURE_ORDER


def log_schema_to_mlflow():
    """
    Loggea el schema y el contrato de features en MLflow.
    Debe llamarse una vez por run de entrenamiento.
    """

    mlflow.log_param("schema_version", SCHEMA_VERSION)
    mlflow.log_param("n_features", len(FEATURE_ORDER))

    schema_artifact = {
        "schema_version": SCHEMA_VERSION,
        "feature_order": FEATURE_ORDER,
        "features": {
            name: {
                "type": dtype.__name__,
                "min": min_v,
                "max": max_v,
            }
            for name, (dtype, min_v, max_v) in FEATURES.items()
        },
    }

    mlflow.log_dict(schema_artifact, "schema.json")
