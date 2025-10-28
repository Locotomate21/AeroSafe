import numpy as np

def evaluate_risk(weather_data: dict) -> dict:
    """
    Evalúa el riesgo de vuelo con base en las condiciones meteorológicas.
    Devuelve una clasificación: Seguro, Precaución o Peligro.
    """

    if not weather_data:
        return {"nivel": "Desconocido", "mensaje": "No hay datos disponibles"}

    temp = weather_data.get("temperatura", 0)
    viento = weather_data.get("viento", 0)
    visibilidad = weather_data.get("visibilidad", 10000)
    humedad = weather_data.get("humedad", 0)
    descripcion = weather_data.get("descripcion", "").lower()

    # --- Regla 1: condiciones severas ---
    if ("tormenta" in descripcion or "lluvia fuerte" in descripcion or "nieve" in descripcion) \
        or viento > 25 or visibilidad < 1500:
        nivel = "Peligro"
        mensaje = "Condiciones severas detectadas. Aterrizaje o vuelo NO recomendado."

    # --- Regla 2: condiciones regulares ---
    elif ("lluvia" in descripcion or "nublado" in descripcion or viento > 10 or visibilidad < 4000 or humedad > 85):
        nivel = "Precaución"
        mensaje = "Condiciones regulares. Se recomienda evaluación adicional."

    # --- Regla 3: condiciones seguras ---
    else:
        nivel = "Seguro"
        mensaje = "Condiciones óptimas para vuelo o aterrizaje."

    return {
        "nivel": nivel,
        "mensaje": mensaje,
        "parametros": {
            "temperatura": temp,
            "viento": viento,
            "visibilidad": visibilidad,
            "humedad": humedad,
            "descripcion": descripcion
        }
    }
