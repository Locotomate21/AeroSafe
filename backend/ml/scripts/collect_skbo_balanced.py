#!/usr/bin/env python3
"""
RECOLECCIÓN DE DATOS REALES - AEROPUERTO SKBO (BOGOTÁ)
Datos reales + sintéticos para balanceo
"""
import pandas as pd
import numpy as np
import asyncio
import requests
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Configuración
sys.path.append(str(Path(__file__).resolve().parents[2]))

SKBO_ICAO = "SKBO"
SKBO_COORDS = {"lat": 4.7016, "lon": -74.1469}
# Sin fallback hardcodeado: la clave viene del entorno (.env) o no hay clave.
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Días de datos históricos a recolectar (OpenWeather Free: solo current + forecast)
# Para históricos reales necesitarías API de pago
DAYS_HISTORY = 7  # Usaremos forecast + current como proxy

# Distribución objetivo
TARGET_DISTRIBUTION = {
    "BAJO": 0.45,      # 45%
    "MODERADO": 0.35,  # 35%
    "ALTO": 0.20       # 20%
}

print("=" * 70)
print("📊 RECOLECCIÓN DE DATOS - AEROPUERTO SKBO BOGOTÁ")
print("=" * 70)
print(f"🎯 Estrategia: Datos reales + sintéticos para balanceo")
print(f"📍 Aeropuerto: {SKBO_ICAO} - El Dorado, Bogotá")
print(f"📅 Período: Actual + Forecast (5 días)")
print("=" * 70)

def get_current_weather_owm(lat, lon, api_key):
    """Obtiene clima actual de OpenWeather"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def get_forecast_owm(lat, lon, api_key):
    """Obtiene pronóstico de OpenWeather"""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def parse_owm_to_features(data, is_current=True):
    """Convierte datos de OpenWeather a features del modelo"""
    
    if is_current:
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        clouds = data.get("clouds", {})
        rain = data.get("rain", {})
        
        features = {
            "temperatura": main.get("temp", 20),
            "humedad": main.get("humidity", 60),
            "presion": main.get("pressure", 1013),
            "viento": wind.get("speed", 0) * 3.6,  # m/s a km/h
            "direccion_viento": wind.get("deg", 0),
            "visibilidad": data.get("visibility", 10000),
            "precipitacion": rain.get("1h", 0) if rain else 0,
            "descripcion": weather.get("main", "Clear").lower(),
            "tipo_nubes": "cubierto" if clouds.get("all", 0) > 75 else "nublado" if clouds.get("all", 0) > 50 else "dispersas" if clouds.get("all", 0) > 25 else "despejado",
            "timestamp": datetime.utcnow().isoformat(),
        }
    else:
        # Forecast item
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        clouds = data.get("clouds", {})
        rain = data.get("rain", {})
        
        features = {
            "temperatura": main.get("temp", 20),
            "humedad": main.get("humidity", 60),
            "presion": main.get("pressure", 1013),
            "viento": wind.get("speed", 0) * 3.6,
            "direccion_viento": wind.get("deg", 0),
            "visibilidad": 10000,  # Forecast no da visibilidad
            "precipitacion": rain.get("3h", 0) / 3 if rain else 0,  # 3h a 1h
            "descripcion": weather.get("main", "Clear").lower(),
            "tipo_nubes": "cubierto" if clouds.get("all", 0) > 75 else "nublado",
            "timestamp": data.get("dt_txt", ""),
        }
    
    return features


def add_aviation_features(features):
    """Agrega features aeronáuticas calculadas"""
    
    # Runway heading de SKBO (13L/31R, 13R/31L - aproximadamente 130°/310°)
    runway_heading = 130  # Pista principal
    
    # Calcular viento cruzado
    wind_dir = features["direccion_viento"]
    wind_speed = features["viento"]
    
    angle_diff = abs(wind_dir - runway_heading)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    crosswind = abs(wind_speed * np.sin(np.radians(angle_diff)))
    headwind = wind_speed * np.cos(np.radians(angle_diff))
    
    # Ráfagas (estimado como 1.3x viento base)
    rafagas = wind_speed * 1.3
    
    # Altitud de densidad (SKBO: 8361 ft = 2548 m)
    altitude_ft = 8361
    temp = features["temperatura"]
    pressure = features["presion"]
    std_temp = 15 - (altitude_ft / 1000 * 2)
    temp_diff = temp - std_temp
    density_alt = altitude_ft + (120 * temp_diff)
    
    # Techo de nubes (estimado según tipo)
    tipo_nubes = features["tipo_nubes"]
    if tipo_nubes == "despejado":
        techo_nubes = 10000
    elif tipo_nubes == "dispersas":
        techo_nubes = 5000
    elif tipo_nubes == "nublado":
        techo_nubes = 2000
    else:  # cubierto
        techo_nubes = 1000
    
    # Condiciones adicionales
    turbulencia = "leve" if wind_speed > 20 else "ninguna"
    estado_pista = "mojada" if features["precipitacion"] > 0 else "seca"
    
    # Riesgo de hielo
    dewpoint = temp - ((100 - features["humedad"]) / 5)  # Aproximado
    riesgo_hielo = (0 <= temp <= 10) and (temp - dewpoint < 3) and features["precipitacion"] > 0
    
    # Fenómenos peligrosos
    tormenta = "thunderstorm" in features["descripcion"] or "storm" in features["descripcion"]
    wind_shear = False  # No disponible en API básica
    
    # Temporales
    now = datetime.utcnow()
    hora = now.hour
    mes = now.month
    dia_año = now.timetuple().tm_yday
    es_noche = hora < 6 or hora > 20
    
    # Agregar todo
    features.update({
        "runway_heading": runway_heading,
        "viento_cruzado": round(crosswind, 1),
        "viento_frente": round(headwind, 1),
        "rafagas": round(rafagas, 1),
        "techo_nubes": techo_nubes,
        "turbulencia": turbulencia,
        "estado_pista": estado_pista,
        "altitud_aeropuerto": altitude_ft,
        "altitud_densidad": round(density_alt, 0),
        "punto_rocio": round(dewpoint, 1),
        "riesgo_hielo": riesgo_hielo,
        "tormenta_electrica": tormenta,
        "cizalladura_viento": wind_shear,
        "hora": hora,
        "mes": mes,
        "dia_año": dia_año,
        "es_noche": es_noche,
    })
    
    return features


def calculate_risk_simple(features):
    """Calcula riesgo de forma simplificada basado en reglas"""
    risk_score = 0
    
    # Factores críticos
    if features["viento_cruzado"] > 35:
        return "ALTO"
    if features["visibilidad"] < 800:
        return "ALTO"
    if features["tormenta_electrica"]:
        return "ALTO"
    
    # Scoring
    if features["viento_cruzado"] > 20:
        risk_score += 20
    elif features["viento_cruzado"] > 15:
        risk_score += 10
    
    if features["visibilidad"] < 3000:
        risk_score += 15
    elif features["visibilidad"] < 5000:
        risk_score += 8
    
    if features["viento"] > 30:
        risk_score += 10
    elif features["viento"] > 20:
        risk_score += 5
    
    if features["precipitacion"] > 10:
        risk_score += 10
    elif features["precipitacion"] > 5:
        risk_score += 5
    
    if features["riesgo_hielo"]:
        risk_score += 10
    
    # Clasificación
    if risk_score >= 40:
        return "ALTO"
    elif risk_score >= 20:
        return "MODERADO"
    else:
        return "BAJO"


def generate_synthetic_sample(base_distribution):
    """Genera muestra sintética para balanceo"""
    from ml.scripts.generate_dataset_UNIFIED import generate_realistic_aviation_weather, calculate_risk_aviation
    
    weather = generate_realistic_aviation_weather()
    risk = calculate_risk_aviation(weather)
    weather["riesgo"] = risk
    weather["source"] = "synthetic"
    
    return weather


# ============================================
# RECOLECCIÓN PRINCIPAL
# ============================================

print("\n📡 Paso 1: Recolectando clima actual de SKBO...")
try:
    current_data = get_current_weather_owm(
        SKBO_COORDS["lat"], 
        SKBO_COORDS["lon"], 
        API_KEY
    )
    current_features = parse_owm_to_features(current_data, is_current=True)
    current_features = add_aviation_features(current_features)
    current_features["riesgo"] = calculate_risk_simple(current_features)
    current_features["source"] = "real_current"
    
    print(f"  ✓ Temperatura: {current_features['temperatura']}°C")
    print(f"  ✓ Viento: {current_features['viento']} km/h")
    print(f"  ✓ Viento cruzado: {current_features['viento_cruzado']} km/h")
    print(f"  ✓ Visibilidad: {current_features['visibilidad']} m")
    print(f"  ✓ Riesgo calculado: {current_features['riesgo']}")
    
    real_samples = [current_features]
except Exception as e:
    print(f"  ✗ Error: {e}")
    real_samples = []

print("\n📡 Paso 2: Recolectando pronóstico de 5 días...")
try:
    forecast_data = get_forecast_owm(
        SKBO_COORDS["lat"], 
        SKBO_COORDS["lon"], 
        API_KEY
    )
    
    for item in forecast_data.get("list", [])[:20]:  # Primeros 20 items (2-3 días)
        features = parse_owm_to_features(item, is_current=False)
        features = add_aviation_features(features)
        features["riesgo"] = calculate_risk_simple(features)
        features["source"] = "real_forecast"
        real_samples.append(features)
    
    print(f"  ✓ {len(real_samples)-1} muestras de pronóstico recolectadas")
except Exception as e:
    print(f"  ✗ Error: {e}")

print(f"\n📊 Total muestras reales: {len(real_samples)}")

# Crear DataFrame con datos reales
real_df = pd.DataFrame(real_samples)

# Analizar distribución de datos reales
print("\n📈 Distribución de datos reales:")
real_distribution = real_df["riesgo"].value_counts()
for risk_class in ["BAJO", "MODERADO", "ALTO"]:
    count = real_distribution.get(risk_class, 0)
    pct = (count / len(real_df) * 100) if len(real_df) > 0 else 0
    print(f"  {risk_class}: {count} ({pct:.1f}%)")

# ============================================
# BALANCEO CON DATOS SINTÉTICOS
# ============================================

print("\n🔄 Paso 3: Balanceando dataset con datos sintéticos...")

# Objetivo: 500 muestras totales balanceadas
TARGET_TOTAL = 500
target_per_class = {
    "BAJO": int(TARGET_TOTAL * TARGET_DISTRIBUTION["BAJO"]),
    "MODERADO": int(TARGET_TOTAL * TARGET_DISTRIBUTION["MODERADO"]),
    "ALTO": int(TARGET_TOTAL * TARGET_DISTRIBUTION["ALTO"]),
}

# Contar cuántas sintéticas necesitamos por clase
synthetic_needed = {}
for risk_class in ["BAJO", "MODERADO", "ALTO"]:
    current_count = real_distribution.get(risk_class, 0)
    needed = max(0, target_per_class[risk_class] - current_count)
    synthetic_needed[risk_class] = needed
    print(f"  {risk_class}: necesitamos {needed} sintéticas")

# Generar sintéticas
print("\n🤖 Generando muestras sintéticas para balanceo...")
synthetic_samples = []
attempts = 0
max_attempts = 10000

current_synthetic_count = {"BAJO": 0, "MODERADO": 0, "ALTO": 0}

while sum(current_synthetic_count.values()) < sum(synthetic_needed.values()) and attempts < max_attempts:
    attempts += 1
    sample = generate_synthetic_sample(TARGET_DISTRIBUTION)
    risk = sample["riesgo"]
    
    if current_synthetic_count[risk] < synthetic_needed[risk]:
        synthetic_samples.append(sample)
        current_synthetic_count[risk] += 1
        
        if len(synthetic_samples) % 50 == 0:
            print(f"  ✓ {len(synthetic_samples)} sintéticas generadas...")

synthetic_df = pd.DataFrame(synthetic_samples)
print(f"  ✓ Total sintéticas: {len(synthetic_df)}")

# ============================================
# COMBINAR Y GUARDAR
# ============================================

print("\n💾 Paso 4: Combinando y guardando dataset final...")

# Combinar
final_df = pd.concat([real_df, synthetic_df], ignore_index=True)

# Mezclar
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar
output_path = "data/dataset/weather_risk_skbo_balanced.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f"  ✓ Dataset guardado: {output_path}")
print(f"  ✓ Total muestras: {len(final_df)}")

# Distribución final
print("\n📊 DISTRIBUCIÓN FINAL:")
final_distribution = final_df["riesgo"].value_counts()
for risk_class in ["BAJO", "MODERADO", "ALTO"]:
    count = final_distribution.get(risk_class, 0)
    pct = (count / len(final_df) * 100)
    real_count = (real_df["riesgo"] == risk_class).sum()
    synth_count = count - real_count
    print(f"  {risk_class}: {count} ({pct:.1f}%) - {real_count} reales + {synth_count} sintéticas")

# Distribución por fuente
print("\n📊 DISTRIBUCIÓN POR FUENTE:")
source_distribution = final_df["source"].value_counts()
for source, count in source_distribution.items():
    pct = (count / len(final_df) * 100)
    print(f"  {source}: {count} ({pct:.1f}%)")

print("\n" + "=" * 70)
print("✅ RECOLECCIÓN COMPLETADA")
print("=" * 70)
print(f"\n📁 Dataset balanceado: {output_path}")
print(f"📊 Total: {len(final_df)} muestras")
print(f"📍 Aeropuerto: SKBO (El Dorado, Bogotá)")
print(f"🎯 Estrategia: {len(real_df)} reales + {len(synthetic_df)} sintéticas para balanceo")
print("\n🚀 Siguiente paso:")
print(f"   python ml/scripts/train_model_mlflow.py --data {output_path}")
print("=" * 70)