"""
GENERADOR DE DATASET AEROSAFE - VERSIÓN UNIFICADA
Genera dataset con 3 clases de riesgo: BAJO, MODERADO, ALTO
30+ variables aeronáuticas realistas

"""
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Se reutilizan el catalogo y la seleccion de cabecera de la aplicacion.
# Duplicar esta logica aqui es exactamente como se genero el desajuste
# entre entrenamiento e inferencia que hubo que corregir.
from features.airports import cabecera_activa, catalogo  # noqa: E402

# Seed para reproducibilidad
np.random.seed(42)
random.seed(42)

# Aeropuertos reales sobre los que se simula. Antes se sorteaban rumbos
# de una lista arbitraria y la altitud de otra, sin relacion entre si.
_AEROPUERTOS = sorted(catalogo().values(), key=lambda a: a.icao)

# Configuración
N_SAMPLES = 5000
OUTPUT_PATH = "data/dataset/weather_risk_aviation.csv"

# Catálogos
DESCRIPTIONS = ["despejado", "nublado", "lluvia_ligera", "lluvia_fuerte", 
                "tormenta", "niebla", "nieve", "granizo"]
CLOUD_TYPES = ["despejado", "dispersas", "nublado", "cubierto"]
RUNWAY_CONDITIONS = ["seca", "humeda", "mojada", "contaminada", "nevada"]
TURBULENCE_LEVELS = ["ninguna", "leve", "moderada", "severa"]

# 🔧 UNIFICADO: 3 clases de riesgo
RISK_CLASSES = ["BAJO", "MODERADO", "ALTO"]


def calculate_crosswind(wind_speed, wind_dir, runway_heading):
    """Calcula componente de viento cruzado"""
    angle_diff = abs(wind_dir - runway_heading)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return abs(wind_speed * np.sin(np.radians(angle_diff)))


def calculate_headwind(wind_speed, wind_dir, runway_heading):
    """Calcula componente de viento de frente/cola"""
    angle_diff = abs(wind_dir - runway_heading)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return wind_speed * np.cos(np.radians(angle_diff))


def calculate_density_altitude(temp, pressure, altitude):
    """Calcula altitud de densidad"""
    std_temp = 15 - (altitude / 1000 * 2)
    temp_diff = temp - std_temp
    return altitude + (120 * temp_diff)


def calculate_risk_aviation(weather_data):
    """
    🔧 VERSIÓN UNIFICADA: Clasificación con 3 niveles
    BAJO, MODERADO, ALTO
    """
    # Extraer variables
    temp = weather_data['temperatura']
    hum = weather_data['humedad']
    wind = weather_data['viento']
    vis = weather_data['visibilidad']
    desc = weather_data['descripcion']
    crosswind = weather_data['viento_cruzado']
    gusts = weather_data['rafagas']
    ceiling = weather_data['techo_nubes']
    precip = weather_data['precipitacion']
    pressure = weather_data['presion']
    runway_cond = weather_data['estado_pista']
    turbulence = weather_data['turbulencia']
    wind_shear = weather_data['cizalladura_viento']
    thunderstorm = weather_data['tormenta_electrica']
    icing = weather_data['riesgo_hielo']
    
    # NIVEL 1: CONDICIONES CRÍTICAS - ALTO
    if crosswind > 35:
        return "ALTO"
    if vis < 800:
        return "ALTO"
    if thunderstorm:
        return "ALTO"
    if wind_shear:
        return "ALTO"
    if turbulence == "severa":
        return "ALTO"
    if gusts > 60:
        return "ALTO"
    if wind > 35 and runway_cond in ["mojada", "nevada", "contaminada"]:
        return "ALTO"
    if ceiling < 200 and vis < 1500:
        return "ALTO"
    if icing and 0 <= temp <= 5 and desc in ["lluvia_ligera", "lluvia_fuerte"]:
        return "ALTO"
    if desc == "granizo":
        return "ALTO"
    
    # NIVEL 2: SCORING PARA MODERADO
    risk_score = 0
    
    # Viento cruzado
    if crosswind > 25:
        risk_score += 25
    elif crosswind > 20:
        risk_score += 18
    elif crosswind > 15:
        risk_score += 12
    elif crosswind > 10:
        risk_score += 6
    
    # Visibilidad
    if vis < 1500:
        risk_score += 20
    elif vis < 3000:
        risk_score += 15
    elif vis < 5000:
        risk_score += 10
    elif vis < 8000:
        risk_score += 5
    
    # Techo de nubes
    if ceiling < 300:
        risk_score += 15
    elif ceiling < 500:
        risk_score += 12
    elif ceiling < 1000:
        risk_score += 8
    elif ceiling < 1500:
        risk_score += 4
    
    # Viento
    if wind > 40:
        risk_score += 15
    elif wind > 30:
        risk_score += 10
    elif wind > 20:
        risk_score += 6
    
    # Ráfagas
    gust_factor = gusts - wind
    if gust_factor > 20:
        risk_score += 12
    elif gust_factor > 15:
        risk_score += 8
    elif gust_factor > 10:
        risk_score += 5
    
    # Precipitación
    if precip > 15:
        risk_score += 10
    elif precip > 10:
        risk_score += 7
    elif precip > 5:
        risk_score += 4
    
    # Estado pista
    if runway_cond == "contaminada":
        risk_score += 10
    elif runway_cond == "nevada":
        risk_score += 8
    elif runway_cond == "mojada":
        risk_score += 5
    elif runway_cond == "humeda":
        risk_score += 2
    
    # Turbulencia
    if turbulence == "moderada":
        risk_score += 10
    elif turbulence == "leve":
        risk_score += 5
    
    # Temperatura extrema
    if temp < -5 or temp > 40:
        risk_score += 8
    elif temp < 0 or temp > 38:
        risk_score += 5
    elif temp < 5 or temp > 35:
        risk_score += 3
    
    # Humedad
    if hum > 95:
        risk_score += 5
    elif hum > 90:
        risk_score += 3
    
    # Presión
    if pressure < 980 or pressure > 1030:
        risk_score += 5
    elif pressure < 990 or pressure > 1025:
        risk_score += 3
    
    # Descripción meteorológica
    if desc == "tormenta":
        risk_score += 8
    elif desc == "nieve":
        risk_score += 6
    elif desc == "lluvia_fuerte":
        risk_score += 5
    elif desc == "niebla":
        risk_score += 4
    elif desc == "lluvia_ligera":
        risk_score += 2
    
    # Riesgo hielo
    if icing:
        risk_score += 7
    
    # Clasificación final
    if risk_score >= 50:
        return "ALTO"
    elif risk_score >= 25:
        return "MODERADO"
    else:
        return "BAJO"


def generate_realistic_aviation_weather():
    """Genera datos meteorológicos realistas"""
    desc = random.choice(DESCRIPTIONS)
    
    # Generar variables según descripción
    if desc == "despejado":
        temp = round(random.uniform(15, 35), 1)
        hum = random.randint(25, 55)
        wind = round(random.uniform(0, 20), 1)
        vis = random.randint(8000, 10000)
        precip = 0
        ceiling = random.randint(5000, 15000)
        clouds = "despejado"
        turbulence = random.choice(["ninguna", "leve"])
        runway_cond = "seca"
        thunderstorm = False
        wind_shear = False
    elif desc == "nublado":
        temp = round(random.uniform(12, 28), 1)
        hum = random.randint(50, 75)
        wind = round(random.uniform(5, 25), 1)
        vis = random.randint(6000, 9000)
        precip = round(random.uniform(0, 2), 1)
        ceiling = random.randint(1500, 4000)
        clouds = random.choice(["dispersas", "nublado"])
        turbulence = random.choice(["ninguna", "leve", "leve"])
        runway_cond = random.choice(["seca", "humeda"])
        thunderstorm = False
        wind_shear = False
    elif desc == "lluvia_ligera":
        temp = round(random.uniform(8, 22), 1)
        hum = random.randint(70, 90)
        wind = round(random.uniform(10, 30), 1)
        vis = random.randint(3000, 7000)
        precip = round(random.uniform(2, 8), 1)
        ceiling = random.randint(800, 2500)
        clouds = random.choice(["nublado", "cubierto"])
        turbulence = random.choice(["leve", "leve", "moderada"])
        runway_cond = random.choice(["humeda", "mojada"])
        thunderstorm = random.random() < 0.1
        wind_shear = random.random() < 0.05
    elif desc == "lluvia_fuerte":
        temp = round(random.uniform(5, 20), 1)
        hum = random.randint(85, 100)
        wind = round(random.uniform(20, 45), 1)
        vis = random.randint(1000, 4000)
        precip = round(random.uniform(10, 25), 1)
        ceiling = random.randint(300, 1500)
        clouds = "cubierto"
        turbulence = random.choice(["moderada", "moderada", "severa"])
        runway_cond = random.choice(["mojada", "contaminada"])
        thunderstorm = random.random() < 0.3
        wind_shear = random.random() < 0.15
    elif desc == "tormenta":
        temp = round(random.uniform(5, 18), 1)
        hum = random.randint(88, 100)
        wind = round(random.uniform(30, 55), 1)
        vis = random.randint(800, 3000)
        precip = round(random.uniform(15, 35), 1)
        ceiling = random.randint(200, 1000)
        clouds = "cubierto"
        turbulence = random.choice(["severa", "severa", "moderada"])
        runway_cond = random.choice(["mojada", "contaminada"])
        thunderstorm = True
        wind_shear = random.random() < 0.4
    elif desc == "niebla":
        temp = round(random.uniform(5, 18), 1)
        hum = random.randint(92, 100)
        wind = round(random.uniform(0, 12), 1)
        vis = random.randint(500, 2500)
        precip = round(random.uniform(0, 2), 1)
        ceiling = random.randint(100, 800)
        clouds = "cubierto"
        turbulence = "ninguna"
        runway_cond = random.choice(["humeda", "mojada"])
        thunderstorm = False
        wind_shear = False
    elif desc == "nieve":
        temp = round(random.uniform(-8, 5), 1)
        hum = random.randint(75, 95)
        wind = round(random.uniform(10, 35), 1)
        vis = random.randint(1000, 4000)
        precip = round(random.uniform(5, 20), 1)
        ceiling = random.randint(400, 2000)
        clouds = "cubierto"
        turbulence = random.choice(["leve", "moderada"])
        runway_cond = random.choice(["nevada", "contaminada"])
        thunderstorm = False
        wind_shear = random.random() < 0.1
    else:  # granizo
        temp = round(random.uniform(10, 25), 1)
        hum = random.randint(70, 90)
        wind = round(random.uniform(25, 50), 1)
        vis = random.randint(1500, 5000)
        precip = round(random.uniform(10, 30), 1)
        ceiling = random.randint(500, 2000)
        clouds = "cubierto"
        turbulence = "severa"
        runway_cond = random.choice(["mojada", "contaminada"])
        thunderstorm = True
        wind_shear = random.random() < 0.3
    
    # Variables adicionales
    wind_direction = random.randint(0, 359)

    # Se toma un aeropuerto real del catalogo y se elige la cabecera EN
    # USO segun el viento, con la misma funcion que usa la inferencia
    # (features/airports.py).
    #
    # Antes el rumbo se sorteaba de una lista fija, sin relacion con la
    # direccion del viento. Eso hacia que 'viento_frente' quedara
    # simetrico alrededor de cero (media -0.09 en el dataset anterior),
    # cuando en operacion real es casi siempre positivo: las aeronaves
    # aterrizan contra el viento. El modelo aprendia una distribucion de
    # viento de cola que no existe.
    aeropuerto = random.choice(_AEROPUERTOS)
    runway_heading = cabecera_activa(wind_direction, aeropuerto)

    gust_factor = round(random.uniform(1.1, 1.5), 1)
    gusts = round(wind * gust_factor, 1)
    crosswind = round(calculate_crosswind(wind, wind_direction, runway_heading), 1)
    headwind = round(calculate_headwind(wind, wind_direction, runway_heading), 1)
    
    if desc in ["tormenta", "lluvia_fuerte"]:
        pressure = random.randint(980, 1005)
    elif desc == "despejado":
        pressure = random.randint(1010, 1030)
    else:
        pressure = random.randint(1000, 1020)
    
    # La altitud es la del aeropuerto elegido arriba, no un sorteo
    # independiente: antes un aerodromo podia salir con rumbo de una
    # pista y elevacion de otra.
    altitude = round(aeropuerto.altitud)
    density_alt = round(calculate_density_altitude(temp, pressure, altitude), 0)
    
    if hum > 90:
        dewpoint = temp - random.uniform(0, 2)
    elif hum > 70:
        dewpoint = temp - random.uniform(2, 5)
    else:
        dewpoint = temp - random.uniform(5, 15)
    dewpoint = round(dewpoint, 1)
    
    icing = (0 <= temp <= 10) and (temp - dewpoint < 3) and precip > 0
    hour = random.randint(0, 23)
    is_night = hour < 6 or hour > 20
    day_of_year = random.randint(1, 365)
    month = min((day_of_year // 30) + 1, 12)
    
    return {
        'temperatura': temp,
        'humedad': hum,
        'viento': wind,
        'visibilidad': vis,
        'descripcion': desc,
        'precipitacion': precip,
        'direccion_viento': wind_direction,
        'runway_heading': runway_heading,
        'viento_cruzado': crosswind,
        'viento_frente': headwind,
        'rafagas': gusts,
        'techo_nubes': ceiling,
        'tipo_nubes': clouds,
        'turbulencia': turbulence,
        'estado_pista': runway_cond,
        'presion': pressure,
        'altitud_aeropuerto': altitude,
        'altitud_densidad': density_alt,
        'punto_rocio': dewpoint,
        'riesgo_hielo': icing,
        'tormenta_electrica': thunderstorm,
        'cizalladura_viento': wind_shear,
        'hora': hour,
        'es_noche': is_night,
        'mes': month,
        'dia_año': day_of_year
    }


# ================================================================= 
# GENERACIÓN DEL DATASET
# =================================================================

print("=" * 70)
print("🚀 GENERADOR DE DATASET AEROSAFE - VERSIÓN UNIFICADA")
print("=" * 70)
print(f"📊 Generando {N_SAMPLES} muestras...")
print(f"🎯 Clases de riesgo: {', '.join(RISK_CLASSES)}")

data = []
risk_counts = {risk: 0 for risk in RISK_CLASSES}

# Distribución objetivo: 45% BAJO, 35% MODERADO, 20% ALTO
target_distribution = {
    "BAJO": N_SAMPLES * 0.45,
    "MODERADO": N_SAMPLES * 0.35,
    "ALTO": N_SAMPLES * 0.20
}

attempts = 0
max_attempts = N_SAMPLES * 15

while len(data) < N_SAMPLES and attempts < max_attempts:
    attempts += 1
    weather_data = generate_realistic_aviation_weather()
    risk = calculate_risk_aviation(weather_data)
    
    if risk_counts[risk] >= target_distribution[risk]:
        continue
    
    weather_data['riesgo'] = risk
    data.append(weather_data)
    risk_counts[risk] += 1
    
    if len(data) % 500 == 0:
        print(f"  ✓ {len(data)}/{N_SAMPLES} muestras...")

# Completar si faltan
while len(data) < N_SAMPLES:
    weather_data = generate_realistic_aviation_weather()
    risk = calculate_risk_aviation(weather_data)
    weather_data['riesgo'] = risk
    data.append(weather_data)
    risk_counts[risk] += 1

# Crear DataFrame
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 70)
print("✅ DATASET GENERADO EXITOSAMENTE")
print("=" * 70)
print(f"\n📁 Archivo: {OUTPUT_PATH}")
print(f"📊 Registros: {len(df)}")
print(f"📋 Variables: {len(df.columns)}")

print("\n🎯 DISTRIBUCIÓN DE CLASES:")
for risk_class in RISK_CLASSES:
    count = risk_counts[risk_class]
    pct = (count / len(df)) * 100
    print(f"  {risk_class}: {count} ({pct:.1f}%)")

print("\n📈 ESTADÍSTICAS POR CLASE:")
for risk in RISK_CLASSES:
    subset = df[df["riesgo"] == risk]
    print(f"\n{risk}:")
    print(f"  Temp: {subset['temperatura'].mean():.1f}°C")
    print(f"  Viento: {subset['viento'].mean():.1f} km/h")
    print(f"  Viento Cruzado: {subset['viento_cruzado'].mean():.1f} km/h")
    print(f"  Visibilidad: {subset['visibilidad'].mean():.0f} m")

print("\n" + "=" * 70)
print("🎯 SIGUIENTE PASO: python ml/scripts/train_model_COMPLETE.py")
print("=" * 70)