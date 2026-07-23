"""
Calibracion de probabilidades.

El modelo base, con class_weight='balanced', ordena bien pero sus scores
no son probabilidades: cuando vota 0.6 el evento ocurre el ~10% de las
veces. Estos tests fijan que la calibracion arregla eso sin estropear el
ranking, y protegen las utilidades que lo miden.
"""
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score

from ml.scripts.calibrate_forecast import ece, umbral_optimo_f1


# =========================================================================
# ECE
# =========================================================================

def test_ece_cero_si_perfectamente_calibrado():
    """Si la probabilidad predicha coincide con la frecuencia real, ECE=0."""
    rng = np.random.default_rng(0)
    # Probabilidades uniformes y etiquetas generadas con esa misma prob.
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert ece(y, p) < 0.02


def test_ece_alto_si_sobreconfiado():
    """
    Predicciones sistematicamente infladas: el modelo asigna
    probabilidades altas (0.6-0.9) a eventos que casi nunca ocurren, como
    hace el modelo balanceado real.
    """
    rng = np.random.default_rng(1)
    p = rng.uniform(0.6, 0.9, 10000)
    y = (rng.uniform(0, 1, 10000) < 0.1).astype(int)  # ocurre el 10%
    assert ece(y, p) > 0.5


# =========================================================================
# Umbral optimo
# =========================================================================

def test_umbral_en_rango():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 1000)
    y = (rng.uniform(0, 1, 1000) < p).astype(int)
    u = umbral_optimo_f1(y, p)
    assert 0.0 <= u <= 1.0


# =========================================================================
# Efecto de la calibracion
# =========================================================================

@pytest.fixture(scope="module")
def escenario_desbalanceado():
    """
    Datos con clase minoritaria al ~5% y un modelo balanceado, que es el
    caso real del pronostico. Split temporal implicito: fit / calib / test
    son particiones disjuntas.
    """
    rng = np.random.default_rng(42)
    n = 12000
    X = rng.normal(size=(n, 6))
    # Señal real: la clase depende de una combinacion de features.
    logit = X[:, 0] * 1.5 + X[:, 1] - 2.5
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

    fit = slice(0, 6000)
    cal = slice(6000, 9000)
    test = slice(9000, n)

    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=0,
    ).fit(X[fit], y[fit])

    return modelo, X, y, cal, test


def test_modelo_balanceado_esta_descalibrado(escenario_desbalanceado):
    """
    Premisa del trabajo: class_weight='balanced' produce scores inflados.
    Si esto dejara de cumplirse, la calibracion sobraria.
    """
    modelo, X, y, _, test = escenario_desbalanceado
    p = modelo.predict_proba(X[test])[:, 1]
    # El score medio del modelo balanceado supera con creces la tasa base.
    assert p.mean() > y[test].mean() * 2


def test_calibracion_mejora_brier(escenario_desbalanceado):
    modelo, X, y, cal, test = escenario_desbalanceado

    p_base = modelo.predict_proba(X[test])[:, 1]
    calibrado = CalibratedClassifierCV(FrozenEstimator(modelo), method="sigmoid").fit(X[cal], y[cal])
    p_cal = calibrado.predict_proba(X[test])[:, 1]

    assert brier_score_loss(y[test], p_cal) < brier_score_loss(y[test], p_base)
    assert ece(y[test], p_cal) < ece(y[test], p_base)


def test_calibracion_no_reordena(escenario_desbalanceado):
    """
    La calibracion es monotona: no cambia el orden de las predicciones,
    asi que el ROC-AUC (que solo depende del orden) se conserva.
    """
    modelo, X, y, cal, test = escenario_desbalanceado

    p_base = modelo.predict_proba(X[test])[:, 1]
    calibrado = CalibratedClassifierCV(FrozenEstimator(modelo), method="sigmoid").fit(X[cal], y[cal])
    p_cal = calibrado.predict_proba(X[test])[:, 1]

    auc_base = roc_auc_score(y[test], p_base)
    auc_cal = roc_auc_score(y[test], p_cal)
    assert abs(auc_base - auc_cal) < 0.01
