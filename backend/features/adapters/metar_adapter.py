"""
Adaptador METAR parseado -> vocabulario del schema.

Traduce la salida de METARTAFService._parse_metar (claves en ingles,
unidades imperiales) a las columnas base del modelo (espanol, unidades
del dataset). No completa features ni predice: de eso se encargan
features.defaults y el servicio de pronostico.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

NUDOS_A_KMH = 1.852

# weather_phenomena del METAR -> categoria 'descripcion' del dataset.
# El orden es de mas severo a menos: un METAR puede traer varios codigos.
FENOMENOS = [
    ("TS", "tormenta"),
    ("GR", "granizo"),
    ("SN", "nieve"),
    ("+RA", "lluvia_fuerte"),
    ("SHRA", "lluvia_fuerte"),
    ("FG", "niebla"),
    ("BR", "niebla"),
    ("HZ", "niebla"),
    ("RA", "lluvia_ligera"),
    ("DZ", "lluvia_ligera"),
]


def _humedad_relativa(temp: float, rocio: float) -> float:
    """Humedad relativa (%) desde temperatura y punto de rocio (Magnus)."""
    b, c = 17.625, 243.04
    hr = 100 * math.exp((b * rocio) / (c + rocio) - (b * temp) / (c + temp))
    return min(max(hr, 0.0), 100.0)


def _descripcion(parsed: Dict[str, Any]) -> str:
    fenomenos = " ".join(parsed.get("weather_phenomena", [])).upper()
    for codigo, categoria in FENOMENOS:
        if codigo in fenomenos:
            return categoria
    return "despejado"


def parsed_metar_to_schema(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convierte un METAR parseado en un dict con las columnas base del
    modelo. Devuelve None si faltan las variables imprescindibles.

    Lo que NO puede derivar (rumbo de pista, altitud, viento cruzado,
    features temporales) lo completa despues features.defaults.
    """
    if "temperature_c" not in parsed or "visibility_m" not in parsed:
        return None

    temp = float(parsed["temperature_c"])
    rocio = float(parsed.get("dewpoint_c", temp - 3))

    viento = parsed.get("wind_speed_kt", 0) * NUDOS_A_KMH
    rafagas = parsed.get("wind_gust_kt", parsed.get("wind_speed_kt", 0)) * NUDOS_A_KMH

    # El METAR reporta 'VRB' cuando el viento es variable (tipico con
    # viento flojo): no es un rumbo, se deja que lo impute el pipeline.
    direccion = parsed.get("wind_direction")
    direccion_num = float(direccion) if isinstance(direccion, (int, float)) else None

    # Techo: capa mas baja con altura reportada.
    alturas = [c["height_ft"] for c in parsed.get("clouds", []) if "height_ft" in c]
    techo = float(min(alturas)) if alturas else None

    fila = {
        "temperatura": temp,
        "punto_rocio": rocio,
        "humedad": _humedad_relativa(temp, rocio),
        "viento": float(viento),
        "rafagas": float(rafagas),
        "visibilidad": float(parsed["visibility_m"]),
        "presion": float(parsed.get("qnh_hpa", 1013)),
        "descripcion": _descripcion(parsed),
        # Se conserva el METAR crudo para calcular precip_intensidad.
        "metar": parsed.get("raw", ""),
    }
    if direccion_num is not None:
        fila["direccion_viento"] = direccion_num
    if techo is not None:
        fila["techo_nubes"] = techo

    return fila
