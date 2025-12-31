import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

n = 5000  # Aumentado para mejor generalización

# Catálogos expandidos
descriptions = ["despejado", "nublado", "lluvia_ligera", "lluvia_fuerte", "tormenta", "niebla", "nieve", "granizo"]
cloud_types = ["despejado", "dispersas", "nublado", "cubierto"]
runway_conditions = ["seca", "humeda", "mojada", "contaminada", "nevada"]
turbulence_levels = ["ninguna", "leve", "moderada", "severa"]

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
    """Calcula altitud de densidad (afecta performance)"""
    # Fórmula simplificada
    std_temp = 15 - (altitude / 1000 * 2)  # ISA
    temp_diff = temp - std_temp
    return altitude + (120 * temp_diff)

def calculate_risk_aviation(weather_data):
    """
    Clasificación de riesgo basada en estándares de aviación REAL
    Considera múltiples factores críticos para la seguridad aérea
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
    
    # =================================================================
    # NIVEL 1: CONDICIONES CRÍTICAS - INMEDIATO PELIGRO
    # =================================================================
    
    # Viento cruzado extremo (límite operacional)
    if crosswind > 35:
        return "Peligro"
    
    # Visibilidad crítica (por debajo de mínimos IFR)
    if vis < 800:
        return "Peligro"
    
    # Tormenta eléctrica en el área
    if thunderstorm:
        return "Peligro"
    
    # Wind shear reportado
    if wind_shear:
        return "Peligro"
    
    # Turbulencia severa
    if turbulence == "severa":
        return "Peligro"
    
    # Ráfagas extremas
    if gusts > 60:
        return "Peligro"
    
    # Combinación de viento y pista contaminada
    if wind > 35 and runway_cond in ["mojada", "nevada", "contaminada"]:
        return "Peligro"
    
    # Techo de nubes crítico + baja visibilidad (IMC severo)
    if ceiling < 200 and vis < 1500:
        return "Peligro"
    
    # Condiciones de hielo + temperatura crítica
    if icing and 0 <= temp <= 5 and desc in ["lluvia_ligera", "lluvia_fuerte"]:
        return "Peligro"
    
    # Granizo (siempre peligroso)
    if desc == "granizo":
        return "Peligro"
    
    # =================================================================
    # NIVEL 2: CONDICIONES DE ALTO RIESGO - PRECAUCIÓN
    # =================================================================
    
    risk_score = 0
    
    # Factor: Viento cruzado (0-25 puntos)
    if crosswind > 25:
        risk_score += 25
    elif crosswind > 20:
        risk_score += 18
    elif crosswind > 15:
        risk_score += 12
    elif crosswind > 10:
        risk_score += 6
    
    # Factor: Visibilidad (0-20 puntos)
    if vis < 1500:
        risk_score += 20
    elif vis < 3000:
        risk_score += 15
    elif vis < 5000:
        risk_score += 10
    elif vis < 8000:
        risk_score += 5
    
    # Factor: Techo de nubes (0-15 puntos)
    if ceiling < 300:
        risk_score += 15
    elif ceiling < 500:
        risk_score += 12
    elif ceiling < 1000:
        risk_score += 8
    elif ceiling < 1500:
        risk_score += 4
    
    # Factor: Viento total (0-15 puntos)
    if wind > 40:
        risk_score += 15
    elif wind > 30:
        risk_score += 10
    elif wind > 20:
        risk_score += 6
    
    # Factor: Ráfagas (0-12 puntos)
    gust_factor = gusts - wind
    if gust_factor > 20:
        risk_score += 12
    elif gust_factor > 15:
        risk_score += 8
    elif gust_factor > 10:
        risk_score += 5
    
    # Factor: Precipitación (0-10 puntos)
    if precip > 15:
        risk_score += 10
    elif precip > 10:
        risk_score += 7
    elif precip > 5:
        risk_score += 4
    
    # Factor: Estado de pista (0-10 puntos)
    if runway_cond == "contaminada":
        risk_score += 10
    elif runway_cond == "nevada":
        risk_score += 8
    elif runway_cond == "mojada":
        risk_score += 5
    elif runway_cond == "humeda":
        risk_score += 2
    
    # Factor: Turbulencia (0-10 puntos)
    if turbulence == "moderada":
        risk_score += 10
    elif turbulence == "leve":
        risk_score += 5
    
    # Factor: Temperatura extrema (0-8 puntos)
    if temp < -5 or temp > 40:
        risk_score += 8
    elif temp < 0 or temp > 38:
        risk_score += 5
    elif temp < 5 or temp > 35:
        risk_score += 3
    
    # Factor: Humedad extrema (0-5 puntos)
    if hum > 95:
        risk_score += 5
    elif hum > 90:
        risk_score += 3
    
    # Factor: Presión anormal (0-5 puntos)
    if pressure < 980 or pressure > 1030:
        risk_score += 5
    elif pressure < 990 or pressure > 1025:
        risk_score += 3
    
    # Factor: Condiciones meteorológicas adversas (0-8 puntos)
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
    
    # Factor: Riesgo de hielo (0-7 puntos)
    if icing:
        risk_score += 7
    
    # =================================================================
    # CLASIFICACIÓN FINAL BASADA EN SCORE
    # =================================================================
    
    if risk_score >= 50:
        return "Peligro"
    elif risk_score >= 25:
        return "Precaución"
    else:
        return "Seguro"

def generate_realistic_aviation_weather():
    """
    Genera datos meteorológicos realistas con todas las variables de aviación
    Incluye correlaciones naturales entre variables
    """
    
    # 1. Elegir condición meteorológica base
    desc = random.choice(descriptions)
    
    # 2. Generar variables correlacionadas según la descripción
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
        
    elif desc == "granizo":
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
    
    # 3. Variables adicionales
    
    # Dirección del viento y heading de pista
    wind_direction = random.randint(0, 359)
    runway_heading = random.choice([90, 180, 270, 0, 45, 135, 225, 315])  # Pistas comunes
    
    # Ráfagas (siempre >= viento base)
    gust_factor = round(random.uniform(1.1, 1.5), 1)
    gusts = round(wind * gust_factor, 1)
    
    # Calcular componentes del viento
    crosswind = round(calculate_crosswind(wind, wind_direction, runway_heading), 1)
    headwind = round(calculate_headwind(wind, wind_direction, runway_heading), 1)
    
    # Presión atmosférica
    if desc in ["tormenta", "lluvia_fuerte"]:
        pressure = random.randint(980, 1005)
    elif desc == "despejado":
        pressure = random.randint(1010, 1030)
    else:
        pressure = random.randint(1000, 1020)
    
    # Altitud del aeropuerto
    altitude = random.choice([0, 500, 1000, 2000, 2500])  # Colombia tiene aeropuertos altos
    
    # Calcular altitud de densidad
    density_alt = round(calculate_density_altitude(temp, pressure, altitude), 0)
    
    # Punto de rocío (dewpoint)
    if hum > 90:
        dewpoint = temp - random.uniform(0, 2)
    elif hum > 70:
        dewpoint = temp - random.uniform(2, 5)
    else:
        dewpoint = temp - random.uniform(5, 15)
    dewpoint = round(dewpoint, 1)
    
    # Riesgo de hielo (icing)
    icing = (0 <= temp <= 10) and (temp - dewpoint < 3) and precip > 0
    
    # Hora del día
    hour = random.randint(0, 23)
    is_night = hour < 6 or hour > 20
    
    # Día del año (estacionalidad)
    day_of_year = random.randint(1, 365)
    
    # Mes (para patrones estacionales)
    month = (day_of_year // 30) + 1
    if month > 12:
        month = 12
    
    # Crear diccionario con todos los datos
    weather_data = {
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
    
    return weather_data

# =================================================================
# GENERACIÓN DEL DATASET
# =================================================================

print("=" * 70)
print("🚀 GENERADOR DE DATASET AVANZADO DE AVIACIÓN")
print("=" * 70)
print(f"📊 Generando {n} muestras con variables de aviación real...")

data = []
risk_counts = {"Seguro": 0, "Precaución": 0, "Peligro": 0}

# Objetivo de distribución: 45% Seguro, 35% Precaución, 20% Peligro
target_distribution = {
    "Seguro": n * 0.45,
    "Precaución": n * 0.35,
    "Peligro": n * 0.20
}

attempts = 0
max_attempts = n * 15

while len(data) < n and attempts < max_attempts:
    attempts += 1
    
    weather_data = generate_realistic_aviation_weather()
    risk = calculate_risk_aviation(weather_data)
    
    # Balancear clases
    if risk_counts[risk] >= target_distribution[risk]:
        continue
    
    # Agregar risk al diccionario
    weather_data['riesgo'] = risk
    
    data.append(weather_data)
    risk_counts[risk] += 1
    
    if len(data) % 500 == 0:
        print(f"  ✓ {len(data)}/{n} muestras generadas...")

# Completar si faltan muestras
while len(data) < n:
    weather_data = generate_realistic_aviation_weather()
    risk = calculate_risk_aviation(weather_data)
    weather_data['riesgo'] = risk
    data.append(weather_data)
    risk_counts[risk] += 1

# Crear DataFrame
df = pd.DataFrame(data)

# Mezclar datos
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar
import os
os.makedirs("data/dataset", exist_ok=True)
df.to_csv("data/dataset/weather_risk_advanced.csv", index=False)

# =================================================================
# ESTADÍSTICAS Y VISUALIZACIÓN
# =================================================================

print("\n" + "=" * 70)
print("✅ DATASET GENERADO EXITOSAMENTE")
print("=" * 70)

print(f"\n📁 Archivo: data/dataset/weather_risk_advanced.csv")
print(f"📊 Total de registros: {len(df)}")
print(f"📋 Total de variables: {len(df.columns)}")

print("\n🎯 DISTRIBUCIÓN DE CLASES:")
print(df["riesgo"].value_counts().sort_index())

print("\n📊 PORCENTAJES:")
percentages = df["riesgo"].value_counts(normalize=True).sort_index() * 100
for risk_level, pct in percentages.items():
    print(f"  {risk_level}: {pct:.2f}%")

print("\n📋 VARIABLES INCLUIDAS:")
print("Variables meteorológicas básicas:")
print("  - temperatura, humedad, viento, visibilidad, descripcion")
print("\nVariables de viento avanzadas:")
print("  - direccion_viento, viento_cruzado, viento_frente, rafagas")
print("\nVariables de aviación:")
print("  - techo_nubes, tipo_nubes, turbulencia, cizalladura_viento")
print("\nVariables de pista/aeropuerto:")
print("  - estado_pista, altitud_aeropuerto, runway_heading")
print("\nVariables meteorológicas avanzadas:")
print("  - precipitacion, presion, punto_rocio, riesgo_hielo")
print("\nFenómenos peligrosos:")
print("  - tormenta_electrica, cizalladura_viento")
print("\nVariables temporales:")
print("  - hora, es_noche, mes, dia_año")
print("\nVariables derivadas:")
print("  - altitud_densidad")

print("\n📈 ESTADÍSTICAS POR CLASE DE RIESGO:")
for risk in ["Seguro", "Precaución", "Peligro"]:
    print(f"\n{'=' * 50}")
    print(f"📊 {risk.upper()}")
    print('=' * 50)
    subset = df[df["riesgo"] == risk]
    
    print(f"  Muestras: {len(subset)}")
    print(f"\n  Variables críticas:")
    print(f"    Temperatura: {subset['temperatura'].mean():.1f}°C (±{subset['temperatura'].std():.1f})")
    print(f"    Viento: {subset['viento'].mean():.1f} km/h (±{subset['viento'].std():.1f})")
    print(f"    Viento Cruzado: {subset['viento_cruzado'].mean():.1f} km/h (±{subset['viento_cruzado'].std():.1f})")
    print(f"    Visibilidad: {subset['visibilidad'].mean():.0f} m (±{subset['visibilidad'].std():.0f})")
    print(f"    Techo Nubes: {subset['techo_nubes'].mean():.0f} m (±{subset['techo_nubes'].std():.0f})")
    print(f"    Precipitación: {subset['precipitacion'].mean():.1f} mm/h (±{subset['precipitacion'].std():.1f})")
    
    print(f"\n  Fenómenos peligrosos:")
    print(f"    Tormentas: {subset['tormenta_electrica'].sum()} ({subset['tormenta_electrica'].sum()/len(subset)*100:.1f}%)")
    print(f"    Wind Shear: {subset['cizalladura_viento'].sum()} ({subset['cizalladura_viento'].sum()/len(subset)*100:.1f}%)")
    print(f"    Riesgo Hielo: {subset['riesgo_hielo'].sum()} ({subset['riesgo_hielo'].sum()/len(subset)*100:.1f}%)")

print("\n" + "=" * 70)
print("🎯 SIGUIENTE PASO:")
print("=" * 70)
print("\n  Ejecuta el entrenamiento actualizado:")
print("  python scripts/train_model.py")
print("\n  El modelo ahora usará 30+ variables en lugar de 4")
print("  ¡Deberías ver una mejora significativa en la precisión!")
print("\n" + "=" * 70)