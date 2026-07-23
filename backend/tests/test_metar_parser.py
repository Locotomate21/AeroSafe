"""
Parser de METAR.

Es el punto por donde entrarán los datos meteorológicos reales, así que
un fallo aquí contamina todo lo que viene después sin dar la cara: un
METAR mal parseado produce una predicción perfectamente formada y
equivocada. El parser es puro, no toca la red, y se puede probar entero.
"""
import pytest

from services.metar_taf_service import METARTAFService


@pytest.fixture(scope="module")
def parser():
    return METARTAFService()


# METAR real de El Dorado: viento 040° a 8 kt, visibilidad 9999 m,
# pocas nubes a 2000 ft, 22/14 °C, QNH 1018.
METAR_SKBO = "METAR SKBO 011200Z 04008KT 9999 FEW020 SCT250 22/14 Q1018 NOSIG"

# Condiciones severas: viento 270° a 20 kt con rachas de 35, visibilidad
# 1200 m, tormenta con lluvia fuerte, techo roto a 800 ft.
METAR_TORMENTA = "METAR SKBO 151800Z 27020G35KT 1200 +TSRA BKN008 OVC015 15/14 Q0995"


def test_identifica_estacion_y_tipo(parser):
    r = parser._parse_metar(METAR_SKBO)

    assert r["report_type"] == "METAR"
    assert r["icao"] == "SKBO"
    assert r["raw"] == METAR_SKBO
    assert "parse_error" not in r


def test_viento_simple(parser):
    r = parser._parse_metar(METAR_SKBO)

    assert r["wind_direction"] == 40
    assert r["wind_speed_kt"] == 8


def test_viento_con_rachas(parser):
    """
    Las rachas importan más que el viento medio para el riesgo: una racha
    de 35 kt es lo que desestabiliza una aproximación.
    """
    r = parser._parse_metar(METAR_TORMENTA)

    assert r["wind_direction"] == 270
    assert r["wind_speed_kt"] == 20
    assert r["wind_gust_kt"] == 35


def test_visibilidad(parser):
    r = parser._parse_metar(METAR_SKBO)

    assert r["visibility_m"] == 9999
    assert r["visibility_km"] == pytest.approx(9.999)


def test_visibilidad_reducida(parser):
    r = parser._parse_metar(METAR_TORMENTA)
    assert r["visibility_m"] == 1200


def test_capas_de_nubes(parser):
    r = parser._parse_metar(METAR_SKBO)

    assert len(r["clouds"]) == 2
    assert r["clouds"][0]["coverage"] == "FEW"
    assert r["clouds"][0]["height_ft"] == 2000
    assert r["clouds"][1]["coverage"] == "SCT"
    assert r["clouds"][1]["height_ft"] == 25000


def test_cumulonimbus_se_identifica(parser):
    """
    'type' se reserva para nubes peligrosas: un CB implica tormenta,
    turbulencia severa y engelamiento, y hay que poder distinguirlo de
    una capa rota cualquiera.
    """
    r = parser._parse_metar("METAR SKBO 011200Z 04008KT 9999 BKN015CB 22/14 Q1018")

    assert r["clouds"][0]["coverage"] == "BKN"
    assert r["clouds"][0]["type"] == "Cumulonimbus"


def test_cielo_despejado(parser):
    r = parser._parse_metar("METAR SKBO 011200Z 04008KT 9999 SKC 22/14 Q1018")
    assert r["clouds"][0]["coverage"] == "SKC"


def test_temperatura_y_punto_de_rocio(parser):
    r = parser._parse_metar(METAR_SKBO)

    assert r["temperature_c"] == 22
    assert r["dewpoint_c"] == 14


def test_temperatura_bajo_cero(parser):
    """En METAR el negativo se escribe con 'M', no con signo menos."""
    r = parser._parse_metar("METAR SKBO 011200Z 04008KT 9999 M02/M05 Q1018")

    assert r["temperature_c"] == -2
    assert r["dewpoint_c"] == -5


def test_qnh_en_hectopascales(parser):
    r = parser._parse_metar(METAR_SKBO)
    assert r["qnh_hpa"] == 1018


def test_qnh_en_pulgadas_se_convierte(parser):
    """Las estaciones de EE.UU. reportan A2992 (29.92 inHg) en vez de Q."""
    r = parser._parse_metar("METAR KJFK 011200Z 04008KT 9999 22/14 A2992")

    assert r["qnh_inhg"] == pytest.approx(29.92)
    assert r["qnh_hpa"] == pytest.approx(1013, abs=2)


def test_fenomenos_meteorologicos(parser):
    r = parser._parse_metar(METAR_TORMENTA)
    assert "+TSRA" in r["weather_phenomena"]


def test_metar_automatico(parser):
    r = parser._parse_metar("METAR SKBO 011200Z AUTO 04008KT 9999 22/14 Q1018")
    assert r["automated"] is True


def test_metar_malformado_no_lanza(parser):
    """
    Un METAR truncado debe devolver lo que se pudo leer, no reventar: la
    red aeronáutica entrega reportes incompletos con más frecuencia de la
    que uno querría.
    """
    r = parser._parse_metar("METAR SKBO")

    assert r["raw"] == "METAR SKBO"
    assert r["icao"] == "SKBO"


def test_metar_vacio_no_lanza(parser):
    r = parser._parse_metar("")
    assert r["raw"] == ""
