"""
Validacion del modelo contra METAR reales.

Por que hace falta: el modelo se entreno con datos sinteticos cuyas
etiquetas produce una funcion de reglas. evaluate_model.py demuestra que
un arbol de profundidad 3 recupera el ~96 % de su rendimiento, asi que la
accuracy sobre ese dataset no dice nada sobre condiciones reales.

Que hace este script, y que NO hace:

  SI  - Descarga METAR reales de aviationweather.gov (NOAA).
  SI  - Los pasa por el pipeline de produccion completo.
  SI  - Compara la prediccion contra los minimos operacionales de
        referencia (RAC / OACI), que son independientes de las reglas
        con las que se etiqueto el dataset.
  SI  - Mide cuanto se aleja la distribucion de los datos reales
        respecto de la de entrenamiento. Un desajuste grande invalida
        cualquier extrapolacion.

  NO  - No mide accuracy contra desenlaces operacionales reales
        (desvios, cancelaciones, go-arounds). Eso exige datos de la
        aerolinea o del explotador del aeropuerto. Sin ellos, ningun
        script puede afirmar que el modelo "acierta".

Uso:
    cd backend
    python -m ml.scripts.validate_with_metar --icao SKBO
    python -m ml.scripts.validate_with_metar --offline   # sin red
"""
import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from features.build_features import NUMERICAL_FEATURES  # noqa: E402
from features.defaults import complete_raw_features  # noqa: E402
from services.metar_taf_service import METARTAFService  # noqa: E402
from services.ml_service_v2 import MLServiceV2  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BACKEND_DIR / "data" / "dataset" / "weather_risk_aviation.csv"

# Minimos meteorologicos de referencia. Son criterios operacionales
# publicados, NO las reglas con las que se genero el dataset: por eso
# sirven como contraste independiente.
#   - ILS CAT I: 550 m de RVR y 200 ft de techo de decision.
#   - VFR: 5000 m de visibilidad y 1500 ft de techo.
MINIMOS = {
    "cat_i_visibilidad_m": 550,
    "cat_i_techo_ft": 200,
    "vfr_visibilidad_m": 5000,
    "vfr_techo_ft": 1500,
}

# METAR reales para poder ejecutar sin red.
METAR_MUESTRA = [
    "METAR SKBO 231200Z 09006KT 9999 SCT020 BKN100 18/12 Q1026 NOSIG",
    "METAR SKBO 231300Z 11008KT 9999 SCT018 BKN080 20/12 Q1025 NOSIG",
    "METAR SKBO 232000Z 18012G22KT 4000 -TSRA BKN012 OVC030 15/13 Q1022",
    "METAR SKBO 232100Z 20015G28KT 1500 TSRA BKN008 OVC020 14/13 Q1020",
    "METAR SKBO 231030Z 00000KT 0500 FG OVC002 11/11 Q1027",
    "METAR SKBO 231100Z VRB02KT 1200 BR BKN003 12/11 Q1027",
    "METAR SKRG 231200Z 18004KT 9999 SCT025 21/14 Q1023 NOSIG",
    "METAR SKCL 231200Z 02010KT 9999 FEW020 SCT100 28/19 Q1011 NOSIG",
]


def riesgo_por_minimos(visibilidad_m: float, techo_ft: float | None) -> str:
    """
    Referencia operacional independiente del modelo.

    No es "la verdad": es el criterio normativo que un despachador
    aplicaria. Sirve para detectar desacuerdos graves, no para calcular
    una accuracy.
    """
    if visibilidad_m < MINIMOS["cat_i_visibilidad_m"]:
        return "ALTO"
    if techo_ft is not None and techo_ft < MINIMOS["cat_i_techo_ft"]:
        return "ALTO"
    if visibilidad_m < 1500:
        return "ALTO"
    if visibilidad_m < MINIMOS["vfr_visibilidad_m"]:
        return "MODERADO"
    if techo_ft is not None and techo_ft < MINIMOS["vfr_techo_ft"]:
        return "MODERADO"
    return "BAJO"


def metar_a_payload(parsed: dict) -> dict | None:
    """Traduce un METAR parseado al vocabulario del modelo."""
    if "temperature_c" not in parsed or "visibility_m" not in parsed:
        return None

    # El METAR reporta el viento en nudos; el modelo se entreno en km/h.
    nudos_a_kmh = 1.852
    viento = parsed.get("wind_speed_kt", 0) * nudos_a_kmh
    rafagas = parsed.get("wind_gust_kt", parsed.get("wind_speed_kt", 0)) * nudos_a_kmh

    capas = parsed.get("clouds", [])
    alturas = [c["height_ft"] for c in capas if "height_ft" in c]
    techo_ft = min(alturas) if alturas else None

    temp = parsed["temperature_c"]
    rocio = parsed.get("dewpoint_c", temp - 3)
    # Humedad relativa desde temperatura y punto de rocio (Magnus inversa).
    b, c = 17.625, 243.04
    import math

    humedad = 100 * math.exp((b * rocio) / (c + rocio) - (b * temp) / (c + temp))

    # El METAR reporta 'VRB' cuando el viento es variable (tipico con
    # viento flojo). No es un rumbo, asi que no se puede calcular
    # componente cruzada: se deja que la impute el pipeline.
    direccion = parsed.get("wind_direction")
    direccion_num = float(direccion) if isinstance(direccion, (int, float)) else None

    payload = {
        "temperatura": float(temp),
        "humedad": float(min(max(humedad, 0), 100)),
        "presion": float(parsed.get("qnh_hpa", 1013)),
        "viento": float(viento),
        "rafagas": float(rafagas),
        "visibilidad": float(parsed["visibility_m"]),
        "punto_rocio": float(rocio),
    }
    if direccion_num is not None:
        payload["direccion_viento"] = direccion_num
    if techo_ft is not None:
        # El dataset usa el techo en pies.
        payload["techo_nubes"] = float(techo_ft)

    fenomenos = " ".join(parsed.get("weather_phenomena", []))
    if "TS" in fenomenos:
        payload["descripcion"] = "tormenta"
        payload["tormenta_electrica"] = 1
    elif "FG" in fenomenos or "BR" in fenomenos:
        payload["descripcion"] = "niebla"
    elif "+RA" in fenomenos or "SHRA" in fenomenos:
        payload["descripcion"] = "lluvia_fuerte"
    elif "RA" in fenomenos or "DZ" in fenomenos:
        payload["descripcion"] = "lluvia_ligera"

    payload["_techo_ft"] = techo_ft
    payload["_raw"] = parsed.get("raw", "")
    return payload


async def descargar_metars(icao: str, horas: int) -> list[str]:
    """Descarga METAR recientes de NOAA."""
    import httpx

    url = "https://aviationweather.gov/api/data/metar"
    params = {"ids": icao, "format": "raw", "hours": str(horas)}

    async with httpx.AsyncClient(timeout=20.0) as cliente:
        respuesta = await cliente.get(url, params=params)
        respuesta.raise_for_status()

    return [l.strip() for l in respuesta.text.splitlines() if l.strip()]


def comparar_distribuciones(reales: pd.DataFrame) -> None:
    """
    Compara los datos reales con la distribucion de entrenamiento.

    Si las condiciones reales caen fuera del rango que el modelo vio, sus
    predicciones son extrapolacion, no interpolacion: el RandomForest no
    extrapola, replica la hoja mas cercana.
    """
    print("\n" + "=" * 72)
    print("DESAJUSTE ENTRE DATOS REALES Y DATOS DE ENTRENAMIENTO")
    print("=" * 72)

    if not DATA_PATH.exists():
        print("  Dataset de entrenamiento no disponible; se omite.")
        return

    entrenamiento = pd.read_csv(DATA_PATH)
    columnas = [c for c in NUMERICAL_FEATURES if c in reales.columns and c in entrenamiento.columns]

    print(f"\n  {'variable':22s} {'real (media)':>14s} {'train (media)':>14s} {'fuera de rango':>16s}")
    print("  " + "-" * 68)

    for col in columnas:
        media_real = reales[col].mean()
        media_train = entrenamiento[col].mean()
        minimo, maximo = entrenamiento[col].min(), entrenamiento[col].max()
        fuera = int(((reales[col] < minimo) | (reales[col] > maximo)).sum())
        marca = f"{fuera}/{len(reales)}" + ("  <-" if fuera else "")
        print(f"  {col:22s} {media_real:>14.1f} {media_train:>14.1f} {marca:>16s}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validacion con METAR reales")
    parser.add_argument("--icao", default="SKBO", help="Codigo ICAO")
    parser.add_argument("--horas", type=int, default=24, help="Horas de historico")
    parser.add_argument("--offline", action="store_true", help="Usar METAR de muestra")
    args = parser.parse_args()

    print("=" * 72)
    print("VALIDACION CON METAR REALES")
    print("=" * 72)

    servicio = MLServiceV2()
    if not servicio.can_infer():
        print("\nERROR: no hay modelo en models/production/. Entrenar primero.")
        return 1

    # ---------- Obtener METAR ----------
    if args.offline:
        crudos = METAR_MUESTRA
        print(f"\n  Modo offline: {len(crudos)} METAR de muestra")
    else:
        try:
            crudos = asyncio.run(descargar_metars(args.icao, args.horas))
            print(f"\n  Descargados {len(crudos)} METAR de {args.icao}")
        except Exception as e:
            print(f"\n  No se pudo descargar ({e}). Usando METAR de muestra.")
            crudos = METAR_MUESTRA

    if not crudos:
        print("  Sin METAR que procesar.")
        return 1

    # ---------- Parsear y predecir ----------
    parser_metar = METARTAFService()
    filas, referencias, brutos = [], [], []

    for crudo in crudos:
        parsed = parser_metar._parse_metar(crudo)
        payload = metar_a_payload(parsed)
        if payload is None:
            continue

        techo = payload.pop("_techo_ft")
        bruto = payload.pop("_raw")
        filas.append(payload)
        referencias.append(riesgo_por_minimos(payload["visibilidad"], techo))
        brutos.append(bruto)

    if not filas:
        print("  Ningun METAR pudo parsearse con los campos necesarios.")
        return 1

    reales = pd.DataFrame(filas)
    resultado = servicio.predict_batch(reales, icao=args.icao)

    # ---------- Comparacion ----------
    print("\n" + "=" * 72)
    print("MODELO vs MINIMOS OPERACIONALES (RAC / OACI)")
    print("=" * 72)
    print(f"\n  {'vis(m)':>8s} {'viento':>8s} {'modelo':>10s} {'conf':>7s} {'minimos':>10s}   {'':3s}")
    print("  " + "-" * 62)

    acuerdos = 0
    subestimaciones = []

    for i in range(len(resultado)):
        prediccion = resultado["riesgo"].iloc[i]
        confianza = resultado["confianza"].iloc[i]
        referencia = referencias[i]
        orden = {"BAJO": 0, "MODERADO": 1, "ALTO": 2}

        if prediccion == referencia:
            marca = "ok"
            acuerdos += 1
        elif orden[prediccion] < orden[referencia]:
            marca = "SUB"  # el modelo subestima el riesgo: lo grave
            subestimaciones.append((brutos[i], prediccion, referencia))
        else:
            marca = "sob"  # sobreestima: conservador, aceptable

        print(
            f"  {reales['visibilidad'].iloc[i]:>8.0f} "
            f"{reales['viento'].iloc[i]:>8.1f} "
            f"{prediccion:>10s} {confianza:>7.3f} {referencia:>10s}   {marca:>3s}"
        )

    total = len(resultado)
    print(f"\n  Coincidencias exactas: {acuerdos}/{total} ({acuerdos / total:.1%})")
    print(f"  Subestimaciones de riesgo: {len(subestimaciones)}/{total}")

    if subestimaciones:
        print("\n  ATENCION - el modelo dio menos riesgo que los minimos operacionales:")
        for bruto, prediccion, referencia in subestimaciones:
            print(f"    {bruto}")
            print(f"      modelo={prediccion}  minimos={referencia}")

    comparar_distribuciones(reales)

    # ---------- Conclusion ----------
    print("\n" + "=" * 72)
    print("ALCANCE DE ESTA VALIDACION")
    print("=" * 72)
    print(
        "\n  Esto NO es una medida de accuracy. Los minimos operacionales son\n"
        "  un criterio normativo de contraste, no el desenlace real de cada\n"
        "  operacion. Para afirmar que el modelo predice bien haria falta el\n"
        "  registro de desvios, cancelaciones y go-arounds del aeropuerto en\n"
        "  las mismas franjas horarias.\n"
        "\n  Lo que si aporta: detecta desacuerdos graves (subestimacion de\n"
        "  riesgo) y cuantifica cuanto se alejan las condiciones reales de la\n"
        "  distribucion con la que se entreno.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
