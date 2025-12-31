import pandas as pd


def openweather_to_raw_df(payload: dict) -> pd.DataFrame:
    """
    Convierte una respuesta JSON de OpenWeather
    en un DataFrame crudo (1 fila).

    No valida schema.
    No construye features.
    No aplica lógica de negocio.
    """

    try:
        row = {
            "temp": payload["main"]["temp"],
            "humidity": payload["main"]["humidity"],
            "pressure": payload["main"]["pressure"],
            "wind_speed": payload["wind"]["speed"],
            "wind_gust": payload.get("wind", {}).get("gust", 0.0),
            "visibility": payload.get("visibility", 0.0),
            "precipitation": _extract_precipitation(payload),
            "clouds": payload.get("clouds", {}).get("all", 0.0),
            "ice_risk": _infer_ice_risk(payload),
        }
    except KeyError as e:
        raise ValueError(f"Payload OpenWeather inválido, falta clave: {e}")

    return pd.DataFrame([row])


def _extract_precipitation(payload: dict) -> float:
    """
    Extrae precipitación en mm si existe.
    OpenWeather puede enviarla en 'rain' o 'snow'.
    """
    rain = payload.get("rain", {}).get("1h", 0.0)
    snow = payload.get("snow", {}).get("1h", 0.0)
    return float(rain) + float(snow)


def _infer_ice_risk(payload: dict) -> int:
    """
    Inferencia mínima y explícita de riesgo de hielo.
    Lógica simple y determinística.
    """
    temp = payload["main"]["temp"]
    precipitation = _extract_precipitation(payload)

    if temp <= 0.0 and precipitation > 0.0:
        return 1
    return 0
