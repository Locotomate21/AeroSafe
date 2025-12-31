import json
from features.schema import SCHEMA_VERSION, FEATURES, FEATURE_ORDER


def validate_model_schema(schema_artifact: dict):
    """
    Valida que el schema del modelo entrenado
    coincida con el schema local del backend.
    """

    # 1. Versión
    if schema_artifact["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version mismatch: "
            f"model={schema_artifact['schema_version']} "
            f"local={SCHEMA_VERSION}"
        )

    # 2. Orden de features
    if schema_artifact["feature_order"] != FEATURE_ORDER:
        raise RuntimeError(
            "Feature order mismatch between model and local schema"
        )

    # 3. Definición de features
    model_features = schema_artifact["features"].keys()
    local_features = FEATURES.keys()

    if set(model_features) != set(local_features):
        raise RuntimeError(
            f"Feature set mismatch: "
            f"model={set(model_features)} "
            f"local={set(local_features)}"
        )
