"""
Catalogo de aeropuertos y seleccion de cabecera activa.

La invariante que protege este fichero: la cabecera en uso nunca produce
componente de viento de cola. Antes el rumbo de pista era una constante
escrita a mano, asi que aproximadamente la mitad de las observaciones
tenian viento de cola, una situacion que en operacion real no se da
porque el aeropuerto cambia de cabecera. El modelo aprendia de ahi una
relacion entre viento y riesgo que no existe.
"""
import math

import pandas as pd
import pytest

from features.airports import (
    Aeropuerto,
    cabecera_activa,
    catalogo,
    diferencia_angular,
    obtener,
)
from features.defaults import complete_raw_features, viento_frente

SKBO = Aeropuerto("SKBO", "El Dorado", 2548.4, 127.0, 307.0)

PAYLOAD = {
    "temperatura": 18.0, "humedad": 70.0, "viento": 20.0,
    "visibilidad": 8000.0, "presion": 1015.0,
}


# =========================================================================
# Catalogo
# =========================================================================

def test_catalogo_tiene_los_internacionales():
    datos = catalogo()
    for icao in ("SKBO", "SKRG", "SKBQ", "SKCL", "SKCG"):
        assert icao in datos, f"falta {icao} en el catalogo"


def test_skbo_tiene_los_valores_reales():
    """
    Regresion: el diccionario escrito a mano tenia rumbo 134 (aproximacion
    magnetica) en vez del rumbo verdadero 127 de OurAirports, que es el
    que se corresponde con la direccion del viento del METAR.
    """
    skbo = obtener("SKBO")

    assert skbo.rumbo_le == pytest.approx(127.0, abs=1.0)
    assert skbo.rumbo_he == pytest.approx(307.0, abs=1.0)
    assert skbo.altitud == pytest.approx(2548, abs=2)


def test_cabeceras_son_opuestas():
    """Las dos cabeceras de una pista difieren en 180 grados."""
    for aeropuerto in catalogo().values():
        assert diferencia_angular(aeropuerto.rumbo_le, aeropuerto.rumbo_he) == pytest.approx(
            180.0, abs=2.0
        ), f"{aeropuerto.icao}: cabeceras no opuestas"


def test_elevaciones_en_metros():
    """
    OurAirports publica la elevacion en PIES; el modelo la usa en metros.
    Regresion: SKCG tenia 4.0 en el diccionario, que eran sus 4 pies
    copiados tal cual como metros.
    """
    assert obtener("SKCG").altitud < 5      # Cartagena, nivel del mar
    assert obtener("SKBO").altitud > 2000   # Bogota, altiplano

    # Ningun aerodromo colombiano supera los 3500 m. Un valor mayor
    # delataria pies sin convertir.
    for aeropuerto in catalogo().values():
        assert 0 <= aeropuerto.altitud < 3500, f"{aeropuerto.icao}: {aeropuerto.altitud}"


def test_icao_desconocido_cae_al_defecto():
    assert obtener("XXXX").icao == "SKBO"
    assert obtener(None).icao == "SKBO"


def test_icao_se_normaliza():
    assert obtener("skbo").icao == "SKBO"
    assert obtener("  SKBO  ").icao == "SKBO"


# =========================================================================
# Cabecera activa
# =========================================================================

@pytest.mark.parametrize("direccion,esperado", [
    (127, 127),   # viento alineado con una cabecera
    (307, 307),   # viento alineado con la opuesta
    (100, 127),
    (330, 307),
    (0, 307),     # el norte esta mas cerca de 307 que de 127
    (180, 127),
])
def test_elige_la_cabecera_contra_el_viento(direccion, esperado):
    assert cabecera_activa(direccion, SKBO) == esperado


def test_nunca_produce_viento_de_cola():
    """
    La invariante central: con la cabecera activa, la componente
    longitudinal del viento es siempre de frente (>= 0).

    Con un rumbo fijo esto fallaba en la mitad del rango de direcciones.
    """
    for direccion in range(0, 360):
        rumbo = cabecera_activa(direccion, SKBO)
        componente = viento_frente(20.0, direccion, rumbo)
        assert componente >= -1e-9, (
            f"viento de cola con direccion {direccion}: "
            f"cabecera {rumbo}, componente {componente:.2f}"
        )


def test_rumbo_fijo_si_produce_viento_de_cola():
    """
    Contraste explicito con el comportamiento anterior, para dejar
    constancia de por que se cambio.
    """
    rumbo_fijo = 134.0
    con_cola = sum(
        1 for direccion in range(0, 360)
        if viento_frente(20.0, direccion, rumbo_fijo) < 0
    )
    assert con_cola > 150, "se esperaba viento de cola en ~la mitad de las direcciones"


def test_invariante_en_todos_los_aeropuertos():
    for aeropuerto in catalogo().values():
        for direccion in range(0, 360, 15):
            rumbo = cabecera_activa(direccion, aeropuerto)
            assert viento_frente(20.0, direccion, rumbo) >= -1e-9, (
                f"{aeropuerto.icao} falla con direccion {direccion}"
            )


def test_diferencia_angular_cruza_el_norte():
    """350 y 10 distan 20 grados, no 340."""
    assert diferencia_angular(350, 10) == pytest.approx(20.0)
    assert diferencia_angular(10, 350) == pytest.approx(20.0)


# =========================================================================
# Integracion con el completado de features
# =========================================================================

def test_completado_usa_cabecera_dinamica():
    """El rumbo imputado depende del viento, no es una constante."""
    norte, _ = complete_raw_features(
        pd.DataFrame([{**PAYLOAD, "direccion_viento": 300.0}]), icao="SKBO"
    )
    sur, _ = complete_raw_features(
        pd.DataFrame([{**PAYLOAD, "direccion_viento": 120.0}]), icao="SKBO"
    )

    assert norte["runway_heading"].iloc[0] != sur["runway_heading"].iloc[0]


def test_completado_no_genera_viento_de_cola():
    filas = [{**PAYLOAD, "direccion_viento": float(d)} for d in range(0, 360, 10)]
    df, _ = complete_raw_features(pd.DataFrame(filas), icao="SKBO")

    assert (df["viento_frente"] >= -1e-9).all(), (
        "el completado produjo viento de cola:\n"
        f"{df.loc[df.viento_frente < 0, ['direccion_viento', 'runway_heading', 'viento_frente']]}"
    )


def test_completado_respeta_rumbo_aportado():
    """Si el cliente conoce la pista en uso, se usa la suya."""
    df, imputados = complete_raw_features(
        pd.DataFrame([{**PAYLOAD, "direccion_viento": 300.0, "runway_heading": 90.0}]),
        icao="SKBO",
    )

    assert df["runway_heading"].iloc[0] == 90.0
    assert "runway_heading" not in imputados


def test_viento_cruzado_maximo_con_viento_perpendicular():
    """Coherencia fisica: el cruzado es maximo a 90 grados de la pista."""
    perpendicular, _ = complete_raw_features(
        pd.DataFrame([{**PAYLOAD, "direccion_viento": 37.0}]), icao="SKBO"
    )
    alineado, _ = complete_raw_features(
        pd.DataFrame([{**PAYLOAD, "direccion_viento": 127.0}]), icao="SKBO"
    )

    assert perpendicular["viento_cruzado"].iloc[0] > alineado["viento_cruzado"].iloc[0]
    assert alineado["viento_cruzado"].iloc[0] == pytest.approx(0.0, abs=0.01)
