"""
Comprueba que la API key de OpenWeather funciona.

Util tras rotar la clave: OpenWeather tarda de 10 min a ~2 h en activar
una key nueva, asi que un 401 recien creada no significa que este mal.

Uso:
    cd backend
    python -m scripts.check_openweather
    python -m scripts.check_openweather --ciudad Medellin,CO
"""
import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica la key de OpenWeather")
    parser.add_argument("--ciudad", default="Bogota,CO")
    args = parser.parse_args()

    key = settings.OPENWEATHER_API_KEY
    if not key:
        print("No hay OPENWEATHER_API_KEY en el entorno. Definirla en backend/.env.")
        return 1

    print(f"Key cargada (termina en ...{key[-4:]}). Consultando {args.ciudad} ...")
    try:
        r = httpx.get(
            settings.BASE_URL,
            params={"q": args.ciudad, "appid": key, "units": "metric", "lang": "es"},
            timeout=20,
        )
    except httpx.HTTPError as e:
        print(f"Sin red o error de conexion: {e}")
        return 1

    if r.status_code == 200:
        d = r.json()
        print(
            f"OK (200) - key ACTIVA. {args.ciudad}: "
            f"{d['main']['temp']} C, {d['weather'][0]['description']}, "
            f"viento {d['wind']['speed']} m/s"
        )
        return 0
    if r.status_code == 401:
        print(
            "401 - la key aun NO esta activa, o es incorrecta. Si acabas de "
            "rotarla, espera hasta ~2 h y vuelve a probar."
        )
        return 2
    print(f"HTTP {r.status_code}: {r.text[:200]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
