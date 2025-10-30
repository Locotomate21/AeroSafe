import joblib
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MLService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
        return cls._instance
    
    def _load_models(self):
        """Carga modelos ML una sola vez (Singleton)"""
        try:
            base_path = Path(__file__).parent.parent.parent / "ml" / "data" / "models"
            
            self.model = joblib.load(base_path / "risk_model.pkl")
            self.scaler = joblib.load(base_path / "scaler.pkl")
            self.encoder = joblib.load(base_path / "label_encoder.pkl")
            
            logger.info("✅ Modelos ML cargados correctamente")
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            raise
    
    def predict_risk(self, data: dict) -> dict:
        """Predice riesgo basado en datos meteorológicos"""
        try:
            # Extraer features base
            temp = data["temperatura"]
            hum = data["humedad"]
            wind = data["viento"]
            vis = data["visibilidad"]
            
            # Feature engineering (mismo que en entrenamiento)
            features = [
                temp,
                hum,
                wind,
                vis,
                temp * hum,                              # temp_x_hum
                wind * (10000 - vis) / 10000,           # viento_x_vis
                # risk_score
                (wind / 50) * 0.3 + ((100 - hum) / 100) * 0.2 + 
                ((10000 - vis) / 10000) * 0.35 + (abs(temp - 22) / 40) * 0.15,
                1 if wind > 35 else 0,                  # viento_alto
                1 if 20 <= wind <= 35 else 0,           # viento_medio
                1 if vis < 3000 else 0,                 # vis_baja
                1 if 3000 <= vis < 6000 else 0,         # vis_media
                1 if temp < 5 or temp > 35 else 0,      # temp_extrema
                1 if hum > 90 else 0                    # humedad_extrema
            ]
            
            # Convertir a numpy array y escalar
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predicción
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Preparar respuesta
            risk_level = self.encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            probs_dict = {
                clase: float(prob) 
                for clase, prob in zip(self.encoder.classes_, probabilities)
            }
            
            logger.info(f"Predicción: {risk_level} (confianza: {confidence:.2f})")
            
            return {
                "risk_level": risk_level,
                "confidence": confidence,
                "probabilities": probs_dict
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise

# Singleton instance
ml_service = MLService()
