from ml.features.build_features import build_features
from ml.features.adapters.openweather_adapter import openweather_to_raw_df
from backend.models.risk_model_loader import load_model
from backend.core.config import settings


class RiskPredictor:
    def __init__(self):
        self.model = load_model(settings.ML_MODEL_URI)

    def predict_from_openweather(self, payload: dict) -> dict:
        """
        Predicción oficial AeroSafe ML
        """

        # Adapter → raw
        raw_df = openweather_to_raw_df(payload)

        # Features + schema validation
        X = build_features(raw_df)

        # Predicción
        prediction = self.model.predict(X)[0]

        return {
            "riesgo_operacional": int(prediction),
            "modelo": "aerosafe_ml",
        }


# Instancia única
risk_predictor = RiskPredictor()
