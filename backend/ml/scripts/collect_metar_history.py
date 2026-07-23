"""
Descarga el archivo historico de METAR desde el IEM (Iowa State).

Fuente: mesonet.agron.iastate.edu — archivo ASOS/AWOS mundial. Publica,
sin API key y sin cuota, pero con throttling: pedir varios anos seguidos
sin pausa devuelve respuestas vacias en silencio, que es peor que un
error. Por eso se descarga ano a ano, con pausa y verificacion.

Salida: un CSV por estacion con las columnas del schema del modelo, mas
el METAR crudo para poder reparsear sin volver a descargar.

Uso:
    cd backend
    python -m ml.scripts.collect_metar_history --icao SKBO --desde 2005
    python -m ml.scripts.collect_metar_history --icao SKBO --desde 2023 --hasta 2024
"""
import argparse
import io
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.airports import cabecera_activa, obtener as obtener_aeropuerto  # noqa: E402
from features.defaults import (  # noqa: E402
    altitud_densidad,
    riesgo_hielo,
    viento_cruzado,
    viento_frente,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
SALIDA_DIR = BACKEND_DIR / "data" / "metar"

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Pausa entre peticiones. El IEM no publica un limite exacto; con menos
# de 5 s se empiezan a recibir respuestas vacias.
PAUSA_S = 6.0
REINTENTOS = 3

NUDOS_A_KMH = 1.852
MILLAS_A_METROS = 1609.34
PIES_A_METROS = 0.3048

# skyc1..4 -> cobertura; se toma la capa mas baja con cobertura
# significativa (BKN u OVC) como techo de nubes.
COBERTURA_SIGNIFICATIVA = {"BKN", "OVC", "VV"}

# Traduccion de los codigos de fenomeno del METAR a las categorias de
# 'descripcion' del modelo. El orden importa: se evalua de mas severo a
# menos, porque un METAR puede traer varios codigos a la vez.
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


def descargar_ano(icao: str, ano: int) -> pd.DataFrame | None:
    """Descarga un ano de observaciones. None si no hay datos."""
    params = {
        "station": icao,
        "data": "all",
        "year1": ano, "month1": 1, "day1": 1,
        "year2": ano + 1, "month2": 1, "day2": 1,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "report_type": "3",  # solo METAR de rutina
    }

    for intento in range(1, REINTENTOS + 1):
        try:
            respuesta = requests.get(IEM_URL, params=params, timeout=180)
            respuesta.raise_for_status()
            texto = respuesta.text

            if texto.count("\n") <= 1:
                # Respuesta vacia: casi siempre es throttling, no ausencia
                # real de datos. Se reintenta con pausa mas larga antes de
                # concluir que el ano no existe.
                if intento < REINTENTOS:
                    espera = PAUSA_S * (intento + 1)
                    print(f"vacio, reintento en {espera:.0f}s ...", end=" ", flush=True)
                    time.sleep(espera)
                    continue
                return None

            return pd.read_csv(io.StringIO(texto), na_values=["M"], low_memory=False)

        except requests.RequestException as e:
            if intento == REINTENTOS:
                print(f"ERROR: {e}")
                return None
            time.sleep(PAUSA_S * (intento + 1))

    return None


def _techo_nubes(fila) -> float:
    """Altura de la capa mas baja con cobertura significativa, en pies."""
    alturas = []
    for i in (1, 2, 3, 4):
        cobertura = fila.get(f"skyc{i}")
        altura = fila.get(f"skyl{i}")
        if (
            isinstance(cobertura, str)
            and cobertura.strip() in COBERTURA_SIGNIFICATIVA
            and pd.notna(altura)
        ):
            alturas.append(float(altura))
    if alturas:
        return min(alturas)
    # Sin capa significativa: cielo practicamente despejado.
    return 20000.0


def _descripcion(codigos) -> str:
    if not isinstance(codigos, str):
        return "despejado"
    texto = codigos.upper()
    for codigo, categoria in FENOMENOS:
        if codigo in texto:
            return categoria
    return "despejado"


def a_schema(bruto: pd.DataFrame, icao: str) -> pd.DataFrame:
    """
    Traduce las columnas del IEM al vocabulario de features/schema.py.

    El IEM reporta en unidades imperiales (F, nudos, millas, pulgadas de
    mercurio); el modelo trabaja en C, km/h, metros y hPa.
    """
    aeropuerto = obtener_aeropuerto(icao)
    d = pd.DataFrame(index=bruto.index)

    d["timestamp"] = pd.to_datetime(bruto["valid"], errors="coerce")

    d["temperatura"] = (bruto["tmpf"] - 32) * 5 / 9
    d["punto_rocio"] = (bruto["dwpf"] - 32) * 5 / 9
    d["humedad"] = bruto["relh"]
    d["viento"] = bruto["sknt"] * NUDOS_A_KMH
    d["rafagas"] = bruto["gust"].fillna(bruto["sknt"]) * NUDOS_A_KMH
    d["direccion_viento"] = bruto["drct"]
    d["visibilidad"] = bruto["vsby"] * MILLAS_A_METROS
    # alti viene en pulgadas de mercurio; el modelo usa QNH en hPa.
    d["presion"] = bruto["alti"] * 33.8639
    d["precipitacion"] = bruto["p01i"] * 25.4  # pulgadas -> mm

    d["techo_nubes"] = bruto.apply(_techo_nubes, axis=1)
    d["descripcion"] = bruto["wxcodes"].apply(_descripcion)

    d["altitud_aeropuerto"] = aeropuerto.altitud
    d["runway_heading"] = d["direccion_viento"].apply(
        lambda direccion: cabecera_activa(direccion, aeropuerto)
        if pd.notna(direccion) else aeropuerto.rumbo_le
    )

    d["hora"] = d["timestamp"].dt.hour
    d["mes"] = d["timestamp"].dt.month
    d["dia_año"] = d["timestamp"].dt.dayofyear
    d["es_noche"] = ((d["hora"] < 6) | (d["hora"] > 20)).astype(int)

    # Se descartan las filas sin las variables imprescindibles antes de
    # calcular derivadas, para no propagar NaN.
    d = d.dropna(subset=["temperatura", "viento", "visibilidad", "presion"])

    d["viento_cruzado"] = d.apply(
        lambda r: viento_cruzado(r["viento"], r["direccion_viento"], r["runway_heading"])
        if pd.notna(r["direccion_viento"]) else 0.0,
        axis=1,
    )
    d["viento_frente"] = d.apply(
        lambda r: viento_frente(r["viento"], r["direccion_viento"], r["runway_heading"])
        if pd.notna(r["direccion_viento"]) else r["viento"],
        axis=1,
    )
    d["altitud_densidad"] = d.apply(
        lambda r: altitud_densidad(r["temperatura"], r["presion"], r["altitud_aeropuerto"]),
        axis=1,
    )
    d["riesgo_hielo"] = d.apply(
        lambda r: riesgo_hielo(
            r["temperatura"],
            r["punto_rocio"] if pd.notna(r["punto_rocio"]) else r["temperatura"] - 3,
            r["precipitacion"] if pd.notna(r["precipitacion"]) else 0.0,
        ),
        axis=1,
    )

    codigos = bruto.loc[d.index, "wxcodes"].fillna("").astype(str).str.upper()
    d["tormenta_electrica"] = codigos.str.contains("TS").astype(int)
    # El METAR no reporta cizalladura salvo en observaciones especiales.
    # Se deja en 0 y se documenta, en vez de inventar un valor.
    d["cizalladura_viento"] = 0

    d["icao"] = icao
    d["metar"] = bruto.loc[d.index, "metar"]

    return d


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga historico METAR del IEM")
    parser.add_argument("--icao", default="SKBO")
    parser.add_argument("--desde", type=int, default=2005)
    parser.add_argument("--hasta", type=int, default=datetime.now().year)
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    icao = args.icao.upper()
    salida = args.salida or SALIDA_DIR / f"metar_{icao.lower()}_{args.desde}_{args.hasta}.csv"

    print("=" * 72)
    print(f"HISTORICO METAR - {icao}  ({args.desde}-{args.hasta})")
    print("=" * 72)
    print(f"\n  Fuente: IEM ASOS  |  pausa entre anos: {PAUSA_S:.0f}s\n")

    trozos = []
    for ano in range(args.desde, args.hasta + 1):
        print(f"  {ano} ...", end=" ", flush=True)
        bruto = descargar_ano(icao, ano)

        if bruto is None or bruto.empty:
            print("sin datos")
        else:
            trozos.append(bruto)
            print(f"{len(bruto):,} observaciones")

        time.sleep(PAUSA_S)

    if not trozos:
        print("\n  No se obtuvo ningun dato.")
        return 1

    bruto = pd.concat(trozos, ignore_index=True)
    print(f"\n  Total descargado: {len(bruto):,} observaciones")

    print("  Traduciendo al schema del modelo ...", end=" ", flush=True)
    datos = a_schema(bruto, icao)
    print(f"{len(datos):,} utilizables ({len(bruto) - len(datos):,} descartadas por datos faltantes)")

    salida.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(salida, index=False, encoding="utf-8")
    print(f"\n  Escrito en {salida.relative_to(BACKEND_DIR)}")

    # ---------- Resumen ----------
    print("\n" + "=" * 72)
    print("RESUMEN")
    print("=" * 72)
    print(f"\n  Periodo: {datos.timestamp.min()}  ->  {datos.timestamp.max()}")
    print(f"  Observaciones: {len(datos):,}\n")

    print("  Distribucion de condiciones:")
    for categoria, cuenta in datos.descripcion.value_counts().items():
        print(f"    {categoria:16s} {cuenta:8,d}  ({cuenta / len(datos):6.2%})")

    print("\n  Eventos poco frecuentes:")
    for etiqueta, mascara in [
        ("visibilidad < 550 m (CAT I)", datos.visibilidad < 550),
        ("visibilidad < 1500 m", datos.visibilidad < 1500),
        ("visibilidad < 5000 m (VFR)", datos.visibilidad < 5000),
        ("viento > 25 kt", datos.viento > 25 * NUDOS_A_KMH),
        ("tormenta electrica", datos.tormenta_electrica == 1),
    ]:
        n = int(mascara.sum())
        print(f"    {etiqueta:30s} {n:8,d}  ({n / len(datos):6.2%})")

    print(
        "\n  Nota: la clase adversa es rara. Con este desbalance la accuracy\n"
        "  deja de ser informativa (predecir siempre BAJO ya acierta >95%).\n"
        "  Usar recall sobre la clase adversa y PR-AUC.\n"
    )
    print(f"  Siguiente paso:  dvc add {salida.relative_to(BACKEND_DIR)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
