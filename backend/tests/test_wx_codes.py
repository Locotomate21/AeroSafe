"""
Traduccion de codigos de fenomeno del METAR.

Es el sustituto de la precipitacion en milimetros, que SKBO no reporta.
Un error aqui se propaga a todo el dataset sin dar la cara.
"""
import pytest

from features.wx_codes import (
    FUERTE,
    LIGERA,
    MODERADA,
    SIN_PRECIPITACION,
    intensidad_precipitacion,
    obstruccion_visibilidad,
    tiene_tormenta,
)


# =========================================================================
# Intensidad de precipitacion
# =========================================================================

@pytest.mark.parametrize("codigos,esperado", [
    ("-RA", LIGERA),
    ("RA", MODERADA),
    ("+RA", FUERTE),
    ("-DZ", LIGERA),
    ("DZ", MODERADA),
    ("+SN", FUERTE),
    ("SHRA", MODERADA),
    ("-SHRA", LIGERA),
    ("+SHRA", FUERTE),
])
def test_escala_de_intensidad(codigos, esperado):
    assert intensidad_precipitacion(codigos) == esperado


@pytest.mark.parametrize("codigos", ["BR", "FG", "HZ", "FU", "", None, "NOSIG"])
def test_fenomenos_sin_precipitacion(codigos):
    """
    La niebla y la bruma reducen la visibilidad pero no son precipitacion.
    Confundirlas inflaria la variable en un aeropuerto donde la niebla es
    el fenomeno adverso mas frecuente (4.79% de las observaciones).
    """
    assert intensidad_precipitacion(codigos) == SIN_PRECIPITACION


def test_tormenta_con_lluvia_es_al_menos_moderada():
    """
    TSRA no lleva marca de intensidad pero no es lluvia suave: una
    tormenta sobre el aerodromo implica precipitacion significativa.
    """
    assert intensidad_precipitacion("TSRA") == MODERADA
    assert intensidad_precipitacion("+TSRA") == FUERTE


def test_en_las_cercanias_pesa_menos():
    """VC = a 8-16 km del campo, no sobre la pista."""
    assert intensidad_precipitacion("VCSH") == LIGERA
    assert intensidad_precipitacion("VCSH") < intensidad_precipitacion("SHRA")


def test_toma_la_intensidad_maxima():
    """Con varios grupos, manda el peor."""
    assert intensidad_precipitacion("-DZ +RA") == FUERTE
    assert intensidad_precipitacion("+RA BR") == FUERTE
    assert intensidad_precipitacion("VCSH TSRA") == MODERADA


def test_la_escala_es_ordenada():
    valores = [
        intensidad_precipitacion(c) for c in ("BR", "-RA", "RA", "+RA")
    ]
    assert valores == sorted(valores)
    assert valores == [0, 1, 2, 3]


def test_funciona_sobre_el_metar_crudo():
    """Debe aceptar tanto wxcodes como el METAR completo."""
    metar = "METAR SKBO 151800Z 27020G35KT 1200 +TSRA BKN008 OVC015 15/14 Q0995"
    assert intensidad_precipitacion(metar) == FUERTE


def test_entrada_no_textual_no_rompe():
    assert intensidad_precipitacion(float("nan")) == SIN_PRECIPITACION
    assert intensidad_precipitacion(123) == SIN_PRECIPITACION


# =========================================================================
# Tormenta
# =========================================================================

@pytest.mark.parametrize("codigos,esperado", [
    ("TSRA", 1), ("+TSRA", 1), ("TS", 1),
    ("-RA", 0), ("BR", 0), ("", 0), (None, 0),
])
def test_deteccion_de_tormenta(codigos, esperado):
    assert tiene_tormenta(codigos) == esperado


# =========================================================================
# Obstruccion a la visibilidad
# =========================================================================

@pytest.mark.parametrize("codigos,esperado", [
    ("FG", "niebla"),
    ("BR", "bruma"),
    ("HZ", "calima"),
    ("-RA", "ninguna"),
    ("", "ninguna"),
    (None, "ninguna"),
])
def test_obstruccion(codigos, esperado):
    assert obstruccion_visibilidad(codigos) == esperado


def test_niebla_tiene_prioridad_sobre_bruma():
    """
    FG y BR son categorias operacionales distintas (visibilidad < 1000 m
    frente a 1000-5000 m). Si aparecen juntas manda la mas restrictiva.
    """
    assert obstruccion_visibilidad("FG BR") == "niebla"
