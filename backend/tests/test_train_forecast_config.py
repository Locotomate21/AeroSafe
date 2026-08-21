"""
El YAML del experimento 01 tiene que describir lo mismo que los valores
por defecto del script.

Esta prueba existe por la razon de siempre en este repositorio: un fallo
aqui no lanza ninguna excepcion. Si el YAML se desvia de DEFAULTS (un
hiperparametro que se olvida, un corte temporal distinto), el
entrenamiento sigue funcionando y produce OTRO modelo sin avisar, y las
comparaciones entre experimentos dejan de significar nada.

No entrena: comprueba la configuracion resuelta, que es lo unico que
distingue a los dos caminos.
"""
from argparse import Namespace
from pathlib import Path

import pytest

from ml.scripts.train_forecast import (
    DEFAULTS,
    ConfigError,
    resolver_config,
)

CONFIG_01 = (
    Path(__file__).resolve().parents[1]
    / "ml" / "config" / "experiments"
    / "01-forecast-randomforest-skbo-v100-training.yaml"
)


def _args(**kwargs) -> Namespace:
    """Los mismos valores que deja argparse cuando no se pasa nada."""
    base = dict(config=None, icao=None, horizonte=None, corte_test=None,
                no_mlflow=False)
    base.update(kwargs)
    return Namespace(**base)


def test_el_yaml_01_reproduce_los_valores_por_defecto():
    pytest.importorskip("yaml")

    por_defecto = resolver_config(_args())
    desde_yaml = resolver_config(_args(config=str(CONFIG_01)))

    assert desde_yaml == por_defecto


def test_los_hiperparametros_del_yaml_01_son_los_historicos():
    pytest.importorskip("yaml")

    cfg = resolver_config(_args(config=str(CONFIG_01)))

    assert cfg["hiperparametros"] == {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }


def test_el_corte_temporal_sigue_siendo_2023():
    # Es un problema de pronostico: el corte va por ano, no al azar.
    assert DEFAULTS["corte_test"] == 2023
    assert resolver_config(_args())["corte_test"] == 2023


def test_sin_config_los_defaults_no_cambian():
    cfg = resolver_config(_args())

    assert cfg["icao"] == "SKBO"
    assert cfg["horizonte"] == 3
    assert cfg["use_mlflow"] is True
    assert cfg["experiment_name"] == "aerosafe-pronostico"
    assert cfg["run_name"] is None


def test_la_cli_gana_sobre_el_yaml():
    pytest.importorskip("yaml")

    cfg = resolver_config(_args(config=str(CONFIG_01), horizonte=6, icao="SKRG"))

    assert cfg["horizonte"] == 6
    assert cfg["icao"] == "SKRG"
    # Lo que la CLI no menciona sigue viniendo del YAML.
    assert cfg["hiperparametros"]["n_estimators"] == 300


def test_no_mlflow_gana_sobre_use_mlflow_del_yaml():
    pytest.importorskip("yaml")

    cfg = resolver_config(_args(config=str(CONFIG_01), no_mlflow=True))

    assert cfg["use_mlflow"] is False


def test_una_familia_no_soportada_se_rechaza(tmp_path):
    pytest.importorskip("yaml")

    ruta = tmp_path / "99-xgboost.yaml"
    ruta.write_text(
        "model_config:\n  model_family: \"xgboost\"\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="xgboost"):
        resolver_config(_args(config=str(ruta)))


def test_un_config_inexistente_se_rechaza(tmp_path):
    with pytest.raises(ConfigError, match="no existe"):
        resolver_config(_args(config=str(tmp_path / "no-esta.yaml")))
