import requests
import json
import os
import sys
import traceback
from datetime import datetime
from backend.utils.config import OPENWEATHER_API_KEY, BASE_URL
from backend.models.database import SessionLocal, Weather

# === RUTA GLOBAL DE LOG ===
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "log.txt")

# === CIUDADES Y CÓDIGOS DE AEROPUERTO ===
CITIES = [
    ("Bogotá", "CO", "SKBO"),
    ("Medellín", "CO", "SKRG"),
    ("Cali", "CO", "SKCL"),
    ("Cartagena", "CO", "SKCG"),
    ("Barranquilla", "CO", "SKBQ"),
    ("Pereira", "CO", "SKPE"),
]


def log_message(message: str):
    """Guarda mensajes con marca de tiempo en log.txt."""
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def get_weather(city_name, country_code="CO", airport_code=None):
    """Obtiene y guarda datos meteorológicos desde la API de OpenWeather."""
    params = {
        "q": f"{city_name},{country_code}",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "es",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            log_message(f"❌ Error obteniendo clima para {city_name}: {data}")
            return None

        weather_info = {
            "ciudad": city_name,
            "pais": country_code,
            "fecha": datetime.now().isoformat(),
            "temperatura": data["main"]["temp"],
            "humedad": data["main"]["humidity"],
            "presion": data["main"]["pressure"],
            "viento": data["wind"]["speed"],
            "visibilidad": data.get("visibility", 0),
            "descripcion": data["weather"][0]["description"],
            "codigo_aeropuerto": airport_code,
            "timestamp": datetime.now().isoformat(),
        }

        # === Guardar los datos SIEMPRE en la raíz del proyecto ===
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        raw_dir = os.path.join(project_root, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)

        file_path = os.path.join(
            raw_dir,
            f"weather_{city_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        )

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(weather_info, f, ensure_ascii=False, indent=4)

        log_message(f"✅ Datos guardados: {file_path}")
        return weather_info

    except Exception as e:
        log_message(f"⚠️ Error en get_weather({city_name}): {e}")
        return None


def save_weather_to_db(weather_info):
    """Guarda la información meteorológica en la base de datos."""
    session = SessionLocal()
    try:
        weather = Weather(
            city=weather_info["ciudad"],
            temperature=weather_info["temperatura"],
            humidity=weather_info["humedad"],
            description=weather_info["descripcion"],
        )
        session.add(weather)
        session.commit()
        log_message(f"🌤️ Clima guardado en DB: {weather_info['ciudad']}")
    except Exception as e:
        log_message(f"❌ Error al guardar en DB: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    log_message("=============================")
    log_message(f"🚀 Ejecutando AEROSAFE - {datetime.now().strftime('%c')}")

    try:
        for city, code, airport in CITIES:
            data = get_weather(city, code, airport)
            if data:
                save_weather_to_db(data)

        log_message("✅ Ejecución completada sin errores.\n")

    except Exception:
        log_message("❌ Error inesperado durante la ejecución:")
        with open(LOG_PATH, "a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        log_message("----------------------------------------\n")
        sys.exit(1)
