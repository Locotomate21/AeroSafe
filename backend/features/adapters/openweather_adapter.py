"""
Adaptador OpenWeather -> schema canónico de AeroSafe.

Traduce la respuesta de la API al vocabulario de features/schema.py
(español, unidades del dataset). No completa features ni predice: eso es
trabajo de features.defaults y del servicio ML.
"""
from typing import Any, Dict

import pandas as pd

# OpenWeather con units=metric devuelve el viento en m/s; el modelo se
# entrenó con km/h (viento medio 16.4, máximo 54.9). Sin esta conversión
# un viento de 15 m/s (54 km/h, fuerte) entra como 15 km/h (flojo).
MS_A_KMH = 3.6

# weather[0].main -> categoría de 'descripcion' del dataset.
CONDICION_OPENWEATHER = {
    "clear": "despejado",
    "clouds": "nublado",
    "drizzle": "lluvia_ligera",
    "rain": "lluvia_ligera",
    "thunderstorm": "tormenta",
    "snow": "nieve",
    "mist": "niebla",
    "fog": "niebla",
    "haze": "niebla",
    "smoke": "niebla",
    "dust": "niebla",
    "sand": "niebla",
    "ash": "niebla",
    "squall": "lluvia_fuerte",
    "tornado": "tormenta",
}


def openweather_to_raw_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Convierte una respuesta JSON de OpenWeather en un DataFrame de una
    fila con nombres canónicos.

    No valida el schema ni completa las features que faltan.

    Raises:
        ValueError: si el payload no trae los campos obligatorios.
    """
    try:
        viento_ms = float(payload["wind"]["speed"])
        rafaga_ms = float(payload.get("wind", {}).get("gust", 0.0))

        row = {
            "temperatura": float(payload["main"]["temp"]),
            "humedad": float(payload["main"]["humidity"]),
            "presion": float(payload["main"]["pressure"]),
            "viento": viento_ms * MS_A_KMH,
            "rafagas": (rafaga_ms or viento_ms) * MS_A_KMH,
            "direccion_viento": float(payload.get("wind", {}).get("deg", 0.0)),
            "visibilidad": float(payload.get("visibility", 10000.0)),
            "precipitacion": _extract_precipitation(payload),
            "descripcion": _map_condicion(payload),
        }
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Payload OpenWeather inválido: {e}") from e

    return pd.DataFrame([row])


def _extract_precipitation(payload: Dict[str, Any]) -> float:
    """
    Precipitación en mm de la última hora.
    OpenWeather la reporta en 'rain' o en 'snow'.
    """
    rain = payload.get("rain", {}).get("1h", 0.0)
    snow = payload.get("snow", {}).get("1h", 0.0)
    return float(rain) + float(snow)


def _map_condicion(payload: Dict[str, Any]) -> str:
    """Traduce weather[0].main a la categoría del dataset."""
    weather = payload.get("weather") or [{}]
    main = str(weather[0].get("main", "")).strip().lower()
    return CONDICION_OPENWEATHER.get(main, "despejado")
