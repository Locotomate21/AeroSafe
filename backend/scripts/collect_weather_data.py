import time
import json
import os
import csv
from datetime import datetime
from backend.utils.weather_data import get_weather

# Lista de aeropuertos colombianos
AIRPORTS = [
    {"city": "Bogotá", "code": "SKBO"},
    {"city": "Medellín", "code": "SKRG"},
    {"city": "Cali", "code": "SKCL"},
    {"city": "Cartagena", "code": "SKCG"},
    {"city": "Barranquilla", "code": "SKBQ"},
    {"city": "Pereira", "code": "SKPE"},
]

def collect_weather_data():
    dataset_dir = "data/dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    all_data = []
    now = datetime.now().strftime("%Y%m%d_%H%M")

    for airport in AIRPORTS:
        city = airport["city"]
        print(f"Obteniendo datos para {city}...")
        data = get_weather(city)
        if data:
            data["codigo_aeropuerto"] = airport["code"]
            data["timestamp"] = datetime.now(datetime.UTC).isoformat()
            all_data.append(data)
        time.sleep(1)  # evita saturar la API

    # Guardar datos en JSON
    json_filename = f"weather_dataset_{now}.json"
    json_path = os.path.join(dataset_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print(f"[OK] Dataset JSON guardado: {json_path}")

    # Guardar datos en CSV
    csv_filename = f"weather_dataset_{now}.csv"
    csv_path = os.path.join(dataset_dir, csv_filename)

    csv_headers = [
        "timestamp", "codigo_aeropuerto", "ciudad", "pais", "temperatura",
        "humedad", "presion", "viento", "visibilidad", "descripcion"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        for item in all_data:
            writer.writerow({
                "timestamp": item.get("timestamp"),
                "codigo_aeropuerto": item.get("codigo_aeropuerto"),
                "ciudad": item.get("ciudad"),
                "pais": item.get("pais"),
                "temperatura": item.get("temperatura"),
                "humedad": item.get("humedad"),
                "presion": item.get("presion"),
                "viento": item.get("viento"),
                "visibilidad": item.get("visibilidad"),
                "descripcion": item.get("descripcion"),
            })

    print(f"[OK] Dataset CSV guardado: {csv_path}")

if __name__ == "__main__":
    collect_weather_data()
