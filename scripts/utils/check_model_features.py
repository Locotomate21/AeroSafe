from backend.services.ml_service_v2 import ml_service_v2

# Asegúrate de que el modelo ya esté cargado en ml_service_v2.model
if ml_service_v2.model is None:
    raise ValueError("El modelo no está cargado. Primero debes asignarlo a ml_service_v2.model")

# Obtener las columnas que el modelo espera
feature_names = ml_service_v2.model.get_booster().feature_names

print("Columnas que el modelo espera:")
for i, f in enumerate(feature_names, 1):
    print(f"{i}: {f}")
