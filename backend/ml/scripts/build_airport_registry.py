"""
Construye el catalogo de aeropuertos a partir de fuentes publicas.

Reemplaza el diccionario AIRPORTS escrito a mano en features/defaults.py,
que tenia varios valores desviados:

    SKBO rumbo    134.0  ->  127 (verdadero)
    SKCL rumbo     20.0  ->   10
    SKCG rumbo     19.0  ->    2
    SKCG altitud    4.0  ->  1.2 m  (se habian copiado los pies como metros)
    SKRG altitud   2142  ->  2120 m

Fuentes (las tres publicas, sin credenciales):

    OurAirports  airports.csv   ICAO, nombre, tipo, lat/lon, elevacion (ft)
    OurAirports  runways.csv    rumbos VERDADEROS, longitud, superficie
    IEM          CO__ASOS       que aeropuertos tienen archivo METAR

Solo se incluyen aeropuertos que tienen las tres cosas: METAR historico,
pista abierta y rumbo conocido. Sin METAR no hay datos con que entrenar;
sin rumbo no se puede calcular viento cruzado.

Nota sobre rumbos: OurAirports publica el rumbo VERDADERO (degT) y el
METAR reporta la direccion del viento tambien en grados verdaderos. Se
corresponden directamente, no hay que corregir declinacion magnetica.

Nota sobre elevacion: OurAirports la da en PIES. El modelo la usa en
METROS. La conversion se hace aqui, una sola vez.

Uso:
    cd backend
    python -m ml.scripts.build_airport_registry
    python -m ml.scripts.build_airport_registry --pais US --salida otro.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

BACKEND_DIR = Path(__file__).resolve().parents[2]
SALIDA_POR_DEFECTO = BACKEND_DIR / "data" / "airports" / "airports_co.csv"

OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data"
IEM_RED = "https://mesonet.agron.iastate.edu/geojson/network/{pais}__ASOS.geojson"

PIES_A_METROS = 0.3048

# Los pequenos y helipuertos no tienen operacion comercial ni METAR fiable.
TIPOS = {"large_airport", "medium_airport"}


def descargar_csv(nombre: str) -> pd.DataFrame:
    url = f"{OURAIRPORTS}/{nombre}"
    print(f"  descargando {nombre} ...", end=" ", flush=True)
    respuesta = requests.get(url, timeout=120)
    respuesta.raise_for_status()

    ruta_tmp = Path(respuesta.request.path_url).name
    df = pd.read_csv(pd.io.common.StringIO(respuesta.text), low_memory=False)
    print(f"{len(df)} filas")
    return df


def estaciones_iem(pais: str) -> dict[str, str]:
    """ICAO -> fecha de inicio del archivo, para las estaciones con METAR."""
    url = IEM_RED.format(pais=pais.upper())
    print(f"  descargando estaciones IEM de {pais} ...", end=" ", flush=True)
    respuesta = requests.get(url, timeout=60)
    respuesta.raise_for_status()

    datos = respuesta.json()["features"]
    print(f"{len(datos)} estaciones")

    return {
        f["id"]: str(f["properties"].get("archive_begin"))[:10]
        for f in datos
    }


def construir(pais: str) -> pd.DataFrame:
    aeropuertos = descargar_csv("airports.csv")
    pistas = descargar_csv("runways.csv")
    iem = estaciones_iem(pais)

    candidatos = aeropuertos[
        (aeropuertos.iso_country == pais.upper())
        & (aeropuertos.type.isin(TIPOS))
        & (aeropuertos.ident.isin(iem))
    ]
    print(f"\n  {len(candidatos)} aeropuertos {pais} de tipo large/medium con METAR")

    # Pistas utilizables: abiertas y con rumbo conocido.
    utiles = pistas[
        (pistas.closed == 0)
        & pistas.le_heading_degT.notna()
        & pistas.he_heading_degT.notna()
    ]

    filas = []
    sin_pista = []

    for _, aeropuerto in candidatos.iterrows():
        propias = utiles[utiles.airport_ident == aeropuerto["ident"]]
        if propias.empty:
            sin_pista.append(aeropuerto["ident"])
            continue

        # La pista principal es la mas larga: es la que se usa en
        # condiciones adversas y para operacion comercial pesada.
        principal = propias.loc[propias.length_ft.idxmax()]

        elevacion_ft = aeropuerto["elevation_ft"]
        filas.append(
            {
                "icao": aeropuerto["ident"],
                "nombre": aeropuerto["name"],
                "tipo": aeropuerto["type"],
                "municipio": aeropuerto["municipality"],
                "latitud": aeropuerto["latitude_deg"],
                "longitud": aeropuerto["longitude_deg"],
                "elevacion_ft": elevacion_ft,
                "elevacion_m": (
                    round(elevacion_ft * PIES_A_METROS, 1)
                    if pd.notna(elevacion_ft) else None
                ),
                # Las DOS cabeceras. La activa se elige en tiempo de
                # inferencia segun el viento: la aeronave aterriza contra
                # el viento, asi que un rumbo fijo genera vientos de cola
                # espurios la mitad del tiempo.
                "pista": f"{principal.le_ident}/{principal.he_ident}",
                "rumbo_le": round(float(principal.le_heading_degT), 1),
                "rumbo_he": round(float(principal.he_heading_degT), 1),
                "largo_ft": principal.length_ft,
                "superficie": principal.surface,
                "n_pistas": len(propias),
                "metar_desde": iem[aeropuerto["ident"]],
            }
        )

    if sin_pista:
        print(f"  descartados por falta de datos de pista: {sorted(sin_pista)}")

    registro = pd.DataFrame(filas).sort_values(
        ["tipo", "largo_ft"], ascending=[True, False]
    )
    return registro.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Catalogo de aeropuertos")
    parser.add_argument("--pais", default="CO", help="Codigo ISO de pais")
    parser.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    args = parser.parse_args()

    print("=" * 72)
    print("CATALOGO DE AEROPUERTOS")
    print("=" * 72 + "\n")

    try:
        registro = construir(args.pais)
    except requests.RequestException as e:
        print(f"\nERROR de red: {e}")
        return 1

    if registro.empty:
        print("\nNo se obtuvo ningun aeropuerto.")
        return 1

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    registro.to_csv(args.salida, index=False, encoding="utf-8")

    print(f"\n  {len(registro)} aeropuertos escritos en {args.salida.relative_to(BACKEND_DIR)}")
    print(f"  por tipo: {registro.tipo.value_counts().to_dict()}")

    grandes = registro[registro.tipo == "large_airport"]
    if not grandes.empty:
        print("\n  Internacionales principales:\n")
        print(f"    {'ICAO':6s}{'nombre':32s}{'elev_m':>8s}{'rumbos':>12s}{'METAR desde':>14s}")
        print("    " + "-" * 70)
        for _, a in grandes.iterrows():
            print(
                f"    {a.icao:6s}{str(a.nombre)[:30]:32s}{a.elevacion_m:>8.0f}"
                f"{f'{a.rumbo_le:.0f}/{a.rumbo_he:.0f}':>12s}{a.metar_desde:>14s}"
            )

    print(
        "\n  Siguiente paso: versionar con DVC\n"
        f"    dvc add {args.salida.relative_to(BACKEND_DIR)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
