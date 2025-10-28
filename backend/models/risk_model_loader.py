import joblib
import numpy as np

model = joblib.load("data/models/risk_model.pkl")
encoder = joblib.load("data/models/label_encoder.pkl")

def predict_risk(features: dict):
    X = np.array([[features["temperatura"], features["humedad"], features["viento"], features["visibilidad"]]])
    y_pred = model.predict(X)
    return encoder.inverse_transform(y_pred)[0]
