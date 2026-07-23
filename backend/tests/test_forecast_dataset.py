"""
Constructor del dataset de pronostico.

El resultado del proyecto depende de dos invariantes que se prueban aqui:

  1. La etiqueta es la condicion a t+N emparejada por MARCA DE TIEMPO,
     no por posicion de fila. Si se emparejara por indice, un hueco en
     los datos convertiria un "futuro a 3h" en uno a 6 o 20h, y la
     metrica seria mentira.

  2. 'descripcion' NO puede estar entre las features: es la variable de
     la que sale la etiqueta. Incluirla reintroduce la circularidad que
     todo este trabajo busca eliminar.
"""
import numpy as np
import pandas as pd
import pytest

from ml.scripts.build_forecast_dataset import (
    FEATURES_BASE,
    construir,
    es_adverso,
)


def _historico(condiciones, inicio="2020-01-01", freq="1h"):
    """Historico sintetico regular para las pruebas."""
    n = len(condiciones)
    ts = pd.date_range(inicio, periods=n, freq=freq)
    return pd.DataFrame({
        "timestamp": ts,
        "temperatura": np.linspace(10, 20, n),
        "punto_rocio": np.linspace(8, 15, n),
        "humedad": np.linspace(60, 90, n),
        "viento": np.full(n, 10.0),
        "rafagas": np.full(n, 15.0),
        "direccion_viento": np.full(n, 130.0),
        "visibilidad": np.full(n, 9000.0),
        "presion": np.full(n, 1026.0),
        "techo_nubes": np.full(n, 3000.0),
        "viento_cruzado": np.full(n, 5.0),
        "altitud_densidad": np.full(n, 3000.0),
        "hora": ts.hour,
        "mes": ts.month,
        "es_noche": ((ts.hour < 6) | (ts.hour > 20)).astype(int),
        "descripcion": condiciones,
        "metar": ["METAR SKBO" for _ in range(n)],
    })


# =========================================================================
# Emparejamiento temporal
# =========================================================================

def test_etiqueta_es_la_condicion_futura():
    """A t+3h: la fila 0 debe tomar como objetivo la condicion de la fila 3."""
    cond = ["despejado", "despejado", "despejado", "niebla",
            "despejado", "despejado", "despejado", "despejado"]
    datos = construir(_historico(cond), horizonte=3)

    # La primera fila (t=00:00) mira a t=03:00, que es 'niebla' -> 1.
    assert datos.iloc[0].objetivo == 1
    # La segunda fila (t=01:00) mira a t=04:00, 'despejado' -> 0.
    assert datos.iloc[1].objetivo == 0


def test_empareja_por_timestamp_no_por_indice():
    """
    Con un hueco en los datos, el emparejamiento debe seguir la marca de
    tiempo. Una fila cuyo t+3h no existe se descarta, no se empareja con
    la siguiente disponible.
    """
    hist = _historico(["despejado"] * 6)
    # Se elimina la observacion de las 03:00; la fila de las 00:00 ya no
    # tiene su par exacto a +3h.
    hist = hist.drop(index=3).reset_index(drop=True)

    datos = construir(hist, horizonte=3)
    tiene_las_00 = (datos.timestamp == pd.Timestamp("2020-01-01 00:00")).any()

    assert not tiene_las_00, "empareja con la fila equivocada tras un hueco"


def test_horizonte_mas_largo_reduce_pares():
    hist = _historico(["despejado"] * 24)
    assert len(construir(hist, horizonte=1)) > len(construir(hist, horizonte=6))


# =========================================================================
# No circularidad
# =========================================================================

def test_descripcion_no_es_feature():
    """La variable de la que sale la etiqueta no puede predecirla."""
    assert "descripcion" not in FEATURES_BASE

    datos = construir(_historico(["despejado"] * 10), horizonte=3)
    assert "descripcion" not in datos.columns


def test_adverso_actual_si_es_feature():
    """
    La persistencia (condicion actual) SI es una feature legitima: es
    informacion disponible en t. Lo que no vale es la condicion futura.
    """
    datos = construir(_historico(["niebla"] + ["despejado"] * 9), horizonte=3)
    assert "adverso_actual" in datos.columns
    assert datos.iloc[0].adverso_actual == 1


# =========================================================================
# Features derivadas
# =========================================================================

def test_hora_ciclica():
    """Las 23h y las 0h deben quedar contiguas en el espacio ciclico."""
    datos = construir(_historico(["despejado"] * 30), horizonte=1)
    # sin^2 + cos^2 = 1 para toda hora.
    suma = datos.hora_sin**2 + datos.hora_cos**2
    assert np.allclose(suma, 1.0)


def test_spread_temperatura_rocio():
    datos = construir(_historico(["despejado"] * 10), horizonte=1)
    esperado = datos.temperatura - datos.punto_rocio
    assert np.allclose(datos.spread_t_td, esperado)


def test_es_adverso():
    s = pd.Series(["niebla", "tormenta", "despejado", "lluvia_ligera"])
    assert es_adverso(s).tolist() == [1, 1, 0, 0]


# =========================================================================
# Integridad
# =========================================================================

def test_sin_nulos_en_features():
    datos = construir(_historico(["despejado", "niebla"] * 15), horizonte=3)
    assert datos[FEATURES_BASE].notna().all().all()


def test_objetivo_es_binario():
    datos = construir(_historico(["despejado", "niebla", "tormenta"] * 10), horizonte=3)
    assert set(datos.objetivo.unique()) <= {0, 1}
