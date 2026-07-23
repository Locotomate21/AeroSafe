# ml/scripts/collect_airport_history.py
"""
Script para recolectar datos históricos de aeropuertos colombianos
"""
import httpx
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

# API Keys (usar la tuya)
OPENWEATHER_API_KEY = "a45d492668dceb132d0d67106b718810"
VISUALCROSSING_API_KEY = ""  # Opcional: registrarse en visualcrossing.com

# Aeropuertos principales de Colombia
AIRPORTS = {
    "SKBO": {"name": "El Dorado", "city": "Bogotá", "lat": 4.7016, "lon": -74.1469},
    "SKCL": {"name": "Alfonso Bonilla", "city": "Cali", "lat": 3.5432, "lon": -76.3816},
    "SKMR": {"name": "Olaya Herrera", "city": "Medellín", "lat": 6.2205, "lon": -75.5909},
    "SKRG": {"name": "José María Córdova", "city": "Rionegro", "lat": 6.1645, "lon": -75.4233},
    "SKCG": {"name": "Rafael Núñez", "city": "Cartagena", "lat": 10.4424, "lon": -75.5130},
    "SKBQ": {"name": "Palonegro", "city": "Bucaramanga", "lat": 7.1265, "lon": -73.1848},
    "SKPE": {"name": "Matecaña", "city": "Pereira", "lat": 4.8127, "lon": -75.7395},
}

async def collect_openweather_history(icao: str, airport: dict, days: int = 5):
    """
    Recolecta datos históricos de OpenWeather (últimos 5 días con plan free)
    """
    print(f"\n📊 Recolectando datos de {airport['name']} ({icao})...")
    
    url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    data_list = []
    
    async with httpx.AsyncClient() as client:
        for day in range(days):
            # Fecha para cada día
            target_date = datetime.now() - timedelta(days=day)
            timestamp = int(target_date.timestamp())
            
            params = {
                "lat": airport["lat"],
                "lon": airport["lon"],
                "dt": timestamp,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            }
            
            try:
                response = await client.get(url, params=params, timeout=15.0)
                
                if response.status_code == 401:
                    print("⚠️ API key inválida o plan no soporta historical")
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                # Procesar datos
                for hour_data in data.get("hourly", []):
                    record = {
                        "icao": icao,
                        "aeropuerto": airport["name"],
                        "ciudad": airport["city"],
                        "timestamp": datetime.fromtimestamp(hour_data["dt"]),
                        "temperatura": hour_data.get("temp", 0),
                        "temp_sensacion": hour_data.get("feels_like", 0),
                        "presion": hour_data.get("pressure", 0),
                        "humedad": hour_data.get("humidity", 0),
                        "punto_rocio": hour_data.get("dew_point", 0),
                        "nubes": hour_data.get("clouds", 0),
                        "uvi": hour_data.get("uvi", 0),
                        "visibilidad": hour_data.get("visibility", 0),
                        "viento": hour_data.get("wind_speed", 0) * 3.6,  # m/s a km/h
                        "viento_direccion": hour_data.get("wind_deg", 0),
                        "viento_rafagas": hour_data.get("wind_gust", 0) * 3.6 if "wind_gust" in hour_data else 0,
                        "lluvia_1h": hour_data.get("rain", {}).get("1h", 0),
                        "nieve_1h": hour_data.get("snow", {}).get("1h", 0),
                        "descripcion": hour_data.get("weather", [{}])[0].get("description", ""),
                        "condicion": hour_data.get("weather", [{}])[0].get("main", ""),
                    }
                    data_list.append(record)
                
                print(f"  ✓ Día {day+1}/{days} completado ({len(hour_data.get('hourly', []))} registros)")
                await asyncio.sleep(1)  # Respetar rate limit
                
            except httpx.HTTPStatusError as e:
                print(f"  ✗ Error HTTP {e.response.status_code}: {e.response.text[:100]}")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
    
    return data_list

async def collect_visualcrossing_history(icao: str, airport: dict, days: int = 30):
    """
    Recolecta datos de Visual Crossing (hasta 1000 llamadas/día gratis)
    Registrarse en: https://www.visualcrossing.com/weather-api
    """
    if not VISUALCROSSING_API_KEY:
        print("⚠️ Visual Crossing API key no configurada")
        return None
    
    print(f"\n📊 Recolectando datos históricos de {airport['name']} ({icao})...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    location = f"{airport['lat']},{airport['lon']}"
    
    params = {
        "key": VISUALCROSSING_API_KEY,
        "unitGroup": "metric",
        "include": "hours",
        "contentType": "json"
    }
    
    full_url = f"{url}/{location}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(full_url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        
        data_list = []
        for day in data.get("days", []):
            for hour in day.get("hours", []):
                record = {
                    "icao": icao,
                    "aeropuerto": airport["name"],
                    "ciudad": airport["city"],
                    "timestamp": datetime.strptime(f"{day['datetime']} {hour['datetime']}", "%Y-%m-%d %H:%M:%S"),
                    "temperatura": hour.get("temp", 0),
                    "temp_sensacion": hour.get("feelslike", 0),
                    "presion": hour.get("pressure", 0),
                    "humedad": hour.get("humidity", 0),
                    "punto_rocio": hour.get("dew", 0),
                    "nubes": hour.get("cloudcover", 0),
                    "visibilidad": hour.get("visibility", 0) * 1000,  # km a metros
                    "viento": hour.get("windspeed", 0),
                    "viento_direccion": hour.get("winddir", 0),
                    "viento_rafagas": hour.get("windgust", 0),
                    "precipitacion": hour.get("precip", 0),
                    "probabilidad_precip": hour.get("precipprob", 0),
                    "descripcion": hour.get("conditions", ""),
                }
                data_list.append(record)
        
        print(f"  ✓ {len(data_list)} registros recolectados")
        return data_list
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return None

async def main():
    """Recolecta datos de todos los aeropuertos"""
    print("=" * 60)
    print("🛫 RECOLECCIÓN DE DATOS HISTÓRICOS DE AEROPUERTOS")
    print("=" * 60)
    
    all_data = []
    
    # Intentar Visual Crossing primero (más completo)
    if VISUALCROSSING_API_KEY:
        print("\n📡 Usando Visual Crossing API (30 días)...")
        for icao, airport in AIRPORTS.items():
            data = await collect_visualcrossing_history(icao, airport, days=30)
            if data:
                all_data.extend(data)
            await asyncio.sleep(2)
    
    # Si no hay Visual Crossing o falló, usar OpenWeather
    if not all_data:
        print("\n📡 Usando OpenWeather API (últimos 5 días)...")
        for icao, airport in AIRPORTS.items():
            data = await collect_openweather_history(icao, airport, days=5)
            if data:
                all_data.extend(data)
            await asyncio.sleep(2)
    
    if not all_data:
        print("\n❌ No se pudieron recolectar datos")
        print("\n💡 Opciones:")
        print("1. Registrarse en Visual Crossing: https://www.visualcrossing.com")
        print("2. Esperar y usar OpenWeather actual (recolectar datos día a día)")
        return
    
    # Crear DataFrame
    df = pd.DataFrame(all_data)
    
    # Guardar datos
    output_dir = Path("../data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_file = output_dir / f"airport_history_{timestamp}.csv"
    json_file = output_dir / f"airport_history_{timestamp}.json"
    
    df.to_csv(csv_file, index=False)
    df.to_json(json_file, orient="records", date_format="iso")
    
    print(f"\n✅ Datos guardados:")
    print(f"   📄 CSV: {csv_file}")
    print(f"   📄 JSON: {json_file}")
    print(f"\n📊 Resumen:")
    print(f"   Total de registros: {len(df)}")
    print(f"   Aeropuertos: {df['icao'].nunique()}")
    print(f"   Período: {df['timestamp'].min()} - {df['timestamp'].max()}")
    print(f"\n📈 Registros por aeropuerto:")
    print(df.groupby('icao')['timestamp'].count())

if __name__ == "__main__":
    asyncio.run(main())