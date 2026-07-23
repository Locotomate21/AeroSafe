"""
Codigos de fenomeno meteorologico del METAR.

Por que existe este modulo: la estacion de SKBO no reporta precipitacion
acumulada (el campo p01i del IEM es cero en las 175.329 observaciones del
archivo). Pero el METAR SI trae la precipitacion, en forma de intensidad
categorica, que ademas es el vocabulario con el que opera la aviacion:
un despachador decide con "lluvia ligera / moderada / fuerte", no con
milimetros por hora.

Se descarto usar reanalisis ERA5 para obtener milimetros: validado contra
el METAR de SKBO en enero de 2024, detecta solo el 32% de las horas con
lluvia observada y el 81% de las que reporta son falsos positivos. Las
celdas de ~25 km de ERA5 no resuelven la lluvia convectiva de la sabana
de Bogota.

Referencia: Anexo 3 OACI, formato de fenomenos presentes.
"""
from __future__ import annotations

import re
from typing import Final

# =========================================================================
# Escala ordinal de intensidad
# =========================================================================
# El METAR marca la intensidad con un prefijo:
#   '-'  ligera      (sin prefijo)  moderada      '+'  fuerte
# 'VC' significa "en las cercanias" (8-16 km del aerodromo): el fenomeno
# no esta sobre el campo, asi que pesa menos que la precipitacion sobre
# la pista, pero mas que nada.

SIN_PRECIPITACION: Final = 0
LIGERA: Final = 1
MODERADA: Final = 2
FUERTE: Final = 3

ETIQUETAS: Final = {
    SIN_PRECIPITACION: "sin_precipitacion",
    LIGERA: "ligera",
    MODERADA: "moderada",
    FUERTE: "fuerte",
}

# Tipos de precipitacion segun el Anexo 3.
#   DZ llovizna   RA lluvia    SN nieve      SG granulos de nieve
#   PL hielo      GR granizo   GS granizo menudo   IC cristales
TIPOS_PRECIPITACION: Final = ("DZ", "RA", "SN", "SG", "PL", "GR", "GS", "IC")

# Descriptores que implican precipitacion aunque no se indique el tipo.
# 'VCSH' (chubascos en las cercanias) es el codigo mas frecuente en SKBO
# —11.83% de las observaciones— y no lleva tipo: SH ya significa
# chubasco, es decir precipitacion. Lo mismo TS sin tipo explicito.
DESCRIPTORES_CON_PRECIPITACION: Final = ("SH", "TS")

# Un grupo de fenomeno es: intensidad opcional, descriptor opcional y
# tipo, donde el tipo puede faltar si el descriptor ya implica
# precipitacion.
# Ej: '-RA', '+TSRA', 'VCSH', 'SHRA', '-DZ', 'TS'
_GRUPO = re.compile(
    r"(?P<intensidad>[+-]|VC)?"
    r"(?P<descriptor>MI|BC|PR|DR|BL|SH|TS|FZ)?"
    r"(?P<tipo>DZ|RA|SN|SG|PL|GR|GS|IC|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)*"
)


def intensidad_precipitacion(codigos: str | None) -> int:
    """
    Traduce los codigos de fenomeno a una escala ordinal 0-3.

    Devuelve la intensidad MAXIMA presente: un METAR puede traer varios
    grupos ('-DZ BR', 'TSRA VCSH') y lo que importa operacionalmente es
    el peor de ellos.

    Args:
        codigos: campo wxcodes del IEM, o el METAR crudo.

    Returns:
        0 sin precipitacion, 1 ligera, 2 moderada, 3 fuerte.

    Ejemplos:
        >>> intensidad_precipitacion("-DZ")
        1
        >>> intensidad_precipitacion("TSRA")
        2
        >>> intensidad_precipitacion("+RA BR")
        3
        >>> intensidad_precipitacion("BR")
        0
    """
    if not isinstance(codigos, str) or not codigos.strip():
        return SIN_PRECIPITACION

    maxima = SIN_PRECIPITACION

    for grupo in _GRUPO.finditer(codigos.upper()):
        tipo = grupo.group("tipo") or ""
        descriptor = grupo.group("descriptor")

        hay_tipo = any(t in tipo for t in TIPOS_PRECIPITACION)
        hay_descriptor = descriptor in DESCRIPTORES_CON_PRECIPITACION

        if not (hay_tipo or hay_descriptor):
            # Fenomeno sin precipitacion: BR (bruma), FG (niebla),
            # HZ (calima)... afectan a la visibilidad, no a la lluvia.
            continue

        intensidad = grupo.group("intensidad")

        if intensidad == "+":
            nivel = FUERTE
        elif intensidad == "-":
            nivel = LIGERA
        elif intensidad == "VC":
            # En las cercanias, no sobre el aerodromo.
            nivel = LIGERA
        else:
            nivel = MODERADA

        # Una tormenta con precipitacion es al menos moderada, aunque el
        # grupo no lleve marca de intensidad.
        if descriptor == "TS" and nivel < MODERADA:
            nivel = MODERADA

        maxima = max(maxima, nivel)

    return maxima


def tiene_tormenta(codigos: str | None) -> int:
    """1 si el METAR reporta actividad tormentosa (TS)."""
    if not isinstance(codigos, str):
        return 0
    return int("TS" in codigos.upper())


def obstruccion_visibilidad(codigos: str | None) -> str:
    """
    Fenomeno que reduce la visibilidad, si lo hay.

    Distinguir niebla (FG, visibilidad < 1000 m) de bruma (BR, 1000-5000 m)
    importa: son categorias operacionales distintas, no matices.
    """
    if not isinstance(codigos, str):
        return "ninguna"

    texto = codigos.upper()
    # De mas restrictivo a menos.
    for codigo, etiqueta in (
        ("FG", "niebla"),
        ("BR", "bruma"),
        ("HZ", "calima"),
        ("FU", "humo"),
        ("DU", "polvo"),
        ("SA", "arena"),
    ):
        if codigo in texto:
            return etiqueta
    return "ninguna"
