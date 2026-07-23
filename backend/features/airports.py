"""
Catalogo de aeropuertos y seleccion de pista activa.

Los datos vienen de data/airports/airports_co.csv, generado por
ml/scripts/build_airport_registry.py a partir de OurAirports e IEM. Ese
fichero se versiona con DVC, no con git, asi que en un clon recien hecho
puede no existir todavia: por eso hay un catalogo minimo embebido como
respaldo.

La parte importante de este modulo no es el catalogo sino
`cabecera_activa()`. Ver su docstring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRO_CSV = BASE_DIR / "data" / "airports" / "airports_co.csv"

# Aeropuerto por defecto cuando no se indica ICAO. AeroSafe se desarrolla
# contra SKBO, asi que asumirlo es menos malo que asumir nivel del mar.
DEFAULT_AIRPORT = "SKBO"


@dataclass(frozen=True)
class Aeropuerto:
    """Un aerodromo con su pista principal."""

    icao: str
    nombre: str
    altitud: float      # metros
    rumbo_le: float     # rumbo verdadero de una cabecera
    rumbo_he: float     # rumbo verdadero de la opuesta


# Respaldo minimo: los seis internacionales, por si falta el CSV.
# Valores tomados del mismo catalogo, no escritos a mano.
_RESPALDO: Dict[str, Aeropuerto] = {
    "SKBO": Aeropuerto("SKBO", "El Dorado", 2548.4, 127.0, 307.0),
    "SKRG": Aeropuerto("SKRG", "Jose Maria Cordova", 2119.9, 360.0, 180.0),
    "SKBQ": Aeropuerto("SKBQ", "Ernesto Cortissoz", 29.9, 40.0, 220.0),
    "SKCL": Aeropuerto("SKCL", "Alfonso Bonilla Aragon", 963.8, 10.0, 190.0),
    "SKCG": Aeropuerto("SKCG", "Rafael Nunez", 1.2, 2.0, 182.0),
    "SKSP": Aeropuerto("SKSP", "Gustavo Rojas Pinilla", 5.8, 59.0, 239.0),
}


@lru_cache(maxsize=1)
def catalogo() -> Dict[str, Aeropuerto]:
    """Carga el catalogo, una sola vez por proceso."""
    if not REGISTRO_CSV.exists():
        logger.warning(
            "No se encuentra %s; se usa el catalogo de respaldo (%d aeropuertos). "
            "Ejecutar 'dvc pull' o 'python -m ml.scripts.build_airport_registry'.",
            REGISTRO_CSV.name,
            len(_RESPALDO),
        )
        return dict(_RESPALDO)

    try:
        df = pd.read_csv(REGISTRO_CSV)
        aeropuertos = {
            fila.icao: Aeropuerto(
                icao=fila.icao,
                nombre=str(fila.nombre),
                altitud=float(fila.elevacion_m),
                rumbo_le=float(fila.rumbo_le),
                rumbo_he=float(fila.rumbo_he),
            )
            for fila in df.itertuples()
        }
        logger.info("Catalogo cargado: %d aeropuertos", len(aeropuertos))
        return aeropuertos
    except Exception as e:
        logger.error("Error leyendo %s (%s); se usa el respaldo", REGISTRO_CSV, e)
        return dict(_RESPALDO)


def obtener(icao: str | None) -> Aeropuerto:
    """Devuelve un aeropuerto, o el por defecto si el ICAO es desconocido."""
    datos = catalogo()
    clave = (icao or DEFAULT_AIRPORT).upper().strip()
    if clave in datos:
        return datos[clave]
    return datos.get(DEFAULT_AIRPORT, _RESPALDO[DEFAULT_AIRPORT])


def diferencia_angular(a: float, b: float) -> float:
    """Angulo agudo entre dos rumbos, en grados (0-180)."""
    return abs(((a - b + 180) % 360) - 180)


def cabecera_activa(direccion_viento: float, aeropuerto: Aeropuerto) -> float:
    """
    Rumbo de la cabecera en uso, dada la direccion del viento.

    Toda pista tiene dos cabeceras opuestas (SKBO: 127 y 307), y las
    aeronaves despegan y aterrizan CONTRA el viento. La cabecera en uso
    es, por tanto, la que forma menor angulo con la direccion de donde
    viene el viento.

    Fijar un unico rumbo —como hacia el diccionario escrito a mano— hace
    que aproximadamente la mitad de las observaciones tengan componente
    de viento de cola, una situacion que en la practica no se da porque
    el aeropuerto simplemente cambia de cabecera. El modelo aprendia asi
    una relacion entre viento y riesgo que no existe.

    Args:
        direccion_viento: grados verdaderos DE DONDE viene el viento.
        aeropuerto: aerodromo con sus dos cabeceras.

    Returns:
        El rumbo verdadero de la cabecera en uso.
    """
    return min(
        (aeropuerto.rumbo_le, aeropuerto.rumbo_he),
        key=lambda rumbo: diferencia_angular(direccion_viento, rumbo),
    )
