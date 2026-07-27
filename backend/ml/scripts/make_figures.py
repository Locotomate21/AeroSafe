"""
Genera las figuras de resultados para la tesis/defensa.

Herramientas: matplotlib + scikit-learn (el stack estandar de ML clasico).
Cada figura se computa desde los datos y modelos versionados, asi que se
regeneran solas si algo cambia. Estilo segun la guia de visualizacion:
fondo claro para impresion, azul secuencial para magnitud, colores de
status reservados, texto en tinta (no en color de serie).

Salida: backend/ml/figures/*.png (y .svg)

Uso:
    cd backend
    python -m ml.scripts.make_figures
    python -m ml.scripts.make_figures --solo fiabilidad transferencia
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.forecast_features import FORECAST_FEATURES  # noqa: E402

BACKEND = Path(__file__).resolve().parents[2]
FDIR = BACKEND / "data" / "forecast"
FIGS = BACKEND / "ml" / "figures"

# --- Paleta (instancia validada, fondo claro para impresion) ---
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"        # secuencial: magnitud / modelo
BLUE_L = "#9ec5f4"
ORANGE = "#eb6834"      # segunda serie
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

CORTE = 2023


def _estilo():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 11, "text.color": INK,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titleweight": "600",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110,
    })


def _guardar(fig, nombre):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"{nombre}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> ml/figures/{nombre}.png")


def _cargar(icao):
    df = pd.read_csv(FDIR / f"forecast_{icao.lower()}_h3.csv",
                     parse_dates=["timestamp"], low_memory=False)
    anio = df.timestamp.dt.year
    return df[anio < CORTE], df[anio >= CORTE]


def _rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42)


# =========================================================================
# 1. Diagrama de fiabilidad (antes / despues de calibrar)
# =========================================================================
def fig_fiabilidad():
    df = pd.read_csv(FDIR / "forecast_skbo_h3.csv", parse_dates=["timestamp"], low_memory=False)
    anio = df.timestamp.dt.year
    tr = df[anio < 2021]; ca = df[(anio >= 2021) & (anio < 2023)]; te = df[anio >= 2023]
    base = _rf().fit(tr[FORECAST_FEATURES].values, tr.objetivo.values)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(
        ca[FORECAST_FEATURES].values, ca.objetivo.values)
    y = te.objetivo.values
    p_sin = base.predict_proba(te[FORECAST_FEATURES].values)[:, 1]
    p_cal = cal.predict_proba(te[FORECAST_FEATURES].values)[:, 1]

    def curva(p):
        d = pd.DataFrame({"y": y, "p": p})
        d["bin"] = pd.qcut(d.p, 10, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(pred=("p", "mean"), real=("y", "mean"))
        return g.pred.values, g.real.values

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.2, label="calibración perfecta", zorder=1)
    x0, r0 = curva(p_sin)
    x1, r1 = curva(p_cal)
    ax.plot(x0, r0, "o-", color=CRITICAL, lw=2, ms=7, label=f"sin calibrar (Brier {brier_score_loss(y,p_sin):.3f})", zorder=3)
    ax.plot(x1, r1, "o-", color=BLUE, lw=2, ms=7, label=f"calibrado (Brier {brier_score_loss(y,p_cal):.3f})", zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia real observada")
    ax.set_title("Diagrama de fiabilidad — SKBO (test 2023-2026)", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.text(0.98, 0.03, "Sin calibrar sobreestima:\ndice 0.6 y ocurre ~0.1.\nCalibrado sigue la diagonal.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK2)
    _guardar(fig, "01_fiabilidad")


# =========================================================================
# 2. Matriz de transferencia (heatmap PR-AUC)
# =========================================================================
def fig_transferencia():
    icaos = ["SKBO", "SKRG", "SKPS", "SKMZ", "SKBQ", "SKCG"]
    icaos = [i for i in icaos if (FDIR / f"forecast_{i.lower()}_h3.csv").exists()]
    datos, modelos, base_rate = {}, {}, {}
    for ic in icaos:
        tr, te = _cargar(ic)
        datos[ic] = (tr, te)
        modelos[ic] = _rf().fit(tr[FORECAST_FEATURES].values, tr.objetivo.values)
        base_rate[ic] = te.objetivo.values.mean()

    M = np.zeros((len(icaos), len(icaos)))
    for i, o in enumerate(icaos):
        for j, d in enumerate(icaos):
            _, te = datos[d]
            p = modelos[o].predict_proba(te[FORECAST_FEATURES].values)[:, 1]
            M[i, j] = average_precision_score(te.objetivo.values, p)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.6, M.max()), aspect="equal")
    ax.set_xticks(range(len(icaos))); ax.set_xticklabels(icaos)
    ax.set_yticks(range(len(icaos))); ax.set_yticklabels(icaos)
    ax.set_xlabel("Evaluado en"); ax.set_ylabel("Entrenado en")
    ax.set_title("Transferencia entre aeropuertos (PR-AUC)", pad=12)
    for i in range(len(icaos)):
        for j in range(len(icaos)):
            col = "#ffffff" if M[i, j] > 0.35 else INK
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold" if i == j else "normal")
            if i == j:  # marco en la celda diagonal (modelo local)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=CRITICAL, lw=2.2))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("PR-AUC", color=INK2); cb.outline.set_visible(False)
    ax.text(0.0, 1.02, "Diagonal enmarcada = modelo local (techo de referencia)",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    _guardar(fig, "02_transferencia")


# =========================================================================
# 3. Diagrama de rendimiento (POD vs FAR por umbral)
# =========================================================================
def fig_rendimiento():
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    colores = {"SKBO": BLUE, "SKRG": ORANGE, "SKPS": GOOD, "SKMZ": CRITICAL}
    for ic, col in colores.items():
        if not (FDIR / f"forecast_{ic.lower()}_h3.csv").exists():
            continue
        tr, te = _cargar(ic)
        m = _rf().fit(tr[FORECAST_FEATURES].values, tr.objetivo.values)
        y = te.objetivo.values
        p = m.predict_proba(te[FORECAST_FEATURES].values)[:, 1]
        pods, fars = [], []
        for u in np.linspace(0.02, 0.9, 40):
            pred = (p >= u).astype(int)
            a = ((pred == 1) & (y == 1)).sum(); b = ((pred == 1) & (y == 0)).sum(); c = ((pred == 0) & (y == 1)).sum()
            pods.append(a / (a + c) if a + c else 0)
            fars.append(b / (a + b) if a + b else np.nan)
        ax.plot(fars, pods, "-", color=col, lw=2, label=f"{ic} (tasa {y.mean():.0%})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("FAR — razón de falsas alarmas  (← mejor)")
    ax.set_ylabel("POD — probabilidad de detección  (mejor →)")
    ax.set_title("Diagrama de rendimiento por aeropuerto", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.text(0.02, 0.97, "No hay punto con POD alta y FAR baja:\nel techo del modelo. Aeropuertos con más\nniebla (SKMZ) tienen mejor curva.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK2)
    _guardar(fig, "03_rendimiento")


# =========================================================================
# 4. Curva de aprendizaje (mas METAR no ayuda)
# =========================================================================
def fig_aprendizaje():
    tr, te = _cargar("SKBO")
    Xte, yte = te[FORECAST_FEATURES].values, te.objetivo.values
    fracs = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    ns, praucs, pos = [], [], []
    for f in fracs:
        n = int(len(tr) * f)
        vals, p = [], []
        for seed in range(3):
            idx = np.random.default_rng(seed).choice(len(tr), n, replace=False)
            sub = tr.iloc[idx]
            if sub.objetivo.sum() < 5:
                continue
            m = _rf().fit(sub[FORECAST_FEATURES].values, sub.objetivo.values)
            vals.append(average_precision_score(yte, m.predict_proba(Xte)[:, 1]))
            p.append(int(sub.objetivo.sum()))
        if vals:
            ns.append(n); praucs.append(np.mean(vals)); pos.append(int(np.mean(p)))

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot(ns, praucs, "o-", color=BLUE, lw=2, ms=7)
    ax.axhline(praucs[-1], ls="--", color=MUTED, lw=1)
    ax.set_xlabel("METAR de entrenamiento")
    ax.set_ylabel("PR-AUC (test 2023-2026)")
    ax.set_title("Curva de aprendizaje — más datos no suben el techo", pad=12)
    ax.set_ylim(min(praucs) - 0.02, max(praucs) + 0.02)
    # anotar eventos positivos en algunos puntos
    for i in (0, 2, len(ns) - 1):
        ax.annotate(f"{pos[i]} eventos", (ns[i], praucs[i]), textcoords="offset points",
                    xytext=(6, -14), fontsize=8, color=INK2)
    ax.text(0.98, 0.05, "Se satura a ~500-1000 eventos.\nDe 45k a 114k METAR: sin cambio.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK2)
    _guardar(fig, "04_aprendizaje")


FIGURAS = {
    "fiabilidad": fig_fiabilidad,
    "transferencia": fig_transferencia,
    "rendimiento": fig_rendimiento,
    "aprendizaje": fig_aprendizaje,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera las figuras de resultados")
    parser.add_argument("--solo", nargs="+", choices=list(FIGURAS), default=list(FIGURAS))
    args = parser.parse_args()

    _estilo()
    print("Generando figuras (matplotlib + scikit-learn)...")
    import warnings
    warnings.filterwarnings("ignore")
    for nombre in args.solo:
        print(f"\n{nombre}:")
        FIGURAS[nombre]()
    print(f"\nListo. Figuras en {FIGS.relative_to(BACKEND)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
