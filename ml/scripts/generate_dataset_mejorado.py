# scripts/generate_dataset_mejorado.py
import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

n = 2000  # Aumentamos a 2000 para mejor entrenamiento

descriptions = ["despejado", "nublado", "lluvia ligera", "lluvia fuerte", "tormenta", "niebla"]

def calculate_risk_improved(temp, hum, wind, vis, desc):
    """
    Función mejorada con lógica clara y realista para clasificar riesgo
    """
    
    # CONDICIONES CRÍTICAS (Prioridad máxima)
    # Si alguna se cumple, automáticamente es PELIGRO
    if wind > 40:  # Vientos muy fuertes
        return "Peligro"
    
    if vis < 1500:  # Visibilidad extremadamente baja
        return "Peligro"
    
    if desc in ["tormenta", "lluvia fuerte"] and wind > 30:
        return "Peligro"
    
    if temp < 0 or temp > 38:  # Temperaturas extremas
        return "Peligro"
    
    if hum > 95 and vis < 3000:  # Niebla densa probable
        return "Peligro"
    
    # CONDICIONES DE ALTO RIESGO (múltiples factores)
    risk_factors = 0
    
    # Factor viento
    if wind > 35:
        risk_factors += 3
    elif wind > 25:
        risk_factors += 2
    elif wind > 15:
        risk_factors += 1
    
    # Factor visibilidad
    if vis < 2000:
        risk_factors += 3
    elif vis < 4000:
        risk_factors += 2
    elif vis < 6000:
        risk_factors += 1
    
    # Factor humedad
    if hum > 90:
        risk_factors += 2
    elif hum > 80:
        risk_factors += 1
    
    # Factor temperatura
    if temp < 5 or temp > 35:
        risk_factors += 2
    elif temp < 10 or temp > 32:
        risk_factors += 1
    
    # Factor descripción climática
    if desc == "tormenta":
        risk_factors += 3
    elif desc == "lluvia fuerte":
        risk_factors += 2
    elif desc == "niebla":
        risk_factors += 2
    elif desc == "lluvia ligera":
        risk_factors += 1
    
    # CLASIFICACIÓN BASADA EN FACTORES DE RIESGO
    if risk_factors >= 6:
        return "Peligro"
    elif risk_factors >= 3:
        return "Precaución"
    else:
        return "Seguro"

def generate_realistic_weather():
    """
    Genera datos meteorológicos más realistas con correlaciones
    """
    # Elegir descripción primero para hacer datos más coherentes
    desc = random.choice(descriptions)
    
    # Generar datos correlacionados según la descripción
    if desc == "despejado":
        temp = round(random.uniform(15, 35), 1)
        hum = random.randint(30, 60)
        wind = round(random.uniform(0, 20), 1)
        vis = random.randint(7000, 10000)
        
    elif desc == "nublado":
        temp = round(random.uniform(10, 30), 1)
        hum = random.randint(50, 80)
        wind = round(random.uniform(5, 25), 1)
        vis = random.randint(5000, 9000)
        
    elif desc == "lluvia ligera":
        temp = round(random.uniform(8, 25), 1)
        hum = random.randint(70, 95)
        wind = round(random.uniform(10, 30), 1)
        vis = random.randint(3000, 7000)
        
    elif desc == "lluvia fuerte":
        temp = round(random.uniform(5, 22), 1)
        hum = random.randint(80, 100)
        wind = round(random.uniform(20, 45), 1)
        vis = random.randint(1500, 5000)
        
    elif desc == "tormenta":
        temp = round(random.uniform(5, 20), 1)
        hum = random.randint(85, 100)
        wind = round(random.uniform(30, 50), 1)
        vis = random.randint(1000, 4000)
        
    elif desc == "niebla":
        temp = round(random.uniform(5, 18), 1)
        hum = random.randint(90, 100)
        wind = round(random.uniform(0, 15), 1)
        vis = random.randint(1000, 3000)
    
    return temp, hum, wind, vis, desc

# --- GENERAR DATASET BALANCEADO ---
print("🔄 Generando dataset mejorado...")

data = []
risk_counts = {"Seguro": 0, "Precaución": 0, "Peligro": 0}

# Generar datos con distribución más balanceada
attempts = 0
max_attempts = n * 10

while len(data) < n and attempts < max_attempts:
    attempts += 1
    
    temp, hum, wind, vis, desc = generate_realistic_weather()
    risk = calculate_risk_improved(temp, hum, wind, vis, desc)
    
    # Balancear clases mientras generamos
    # Queremos aproximadamente: 40% Seguro, 35% Precaución, 25% Peligro
    if risk == "Seguro" and risk_counts["Seguro"] >= n * 0.40:
        continue
    elif risk == "Precaución" and risk_counts["Precaución"] >= n * 0.35:
        continue
    elif risk == "Peligro" and risk_counts["Peligro"] >= n * 0.25:
        continue
    
    data.append([temp, hum, wind, vis, desc, risk])
    risk_counts[risk] += 1

# Si no llegamos al total, completar sin restricciones
while len(data) < n:
    temp, hum, wind, vis, desc = generate_realistic_weather()
    risk = calculate_risk_improved(temp, hum, wind, vis, desc)
    data.append([temp, hum, wind, vis, desc, risk])
    risk_counts[risk] += 1

# Crear DataFrame
df = pd.DataFrame(data, columns=["temperatura", "humedad", "viento", "visibilidad", "descripcion", "riesgo"])

# Mezclar datos
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar
df.to_csv("data/dataset/weather_risk.csv", index=False)

# --- ESTADÍSTICAS ---
print("\n✅ Dataset generado en data/dataset/weather_risk.csv")
print(f"📊 Total de registros: {len(df)}")
print("\n📈 Distribución de clases:")
print(df["riesgo"].value_counts())
print("\n📊 Porcentajes:")
print(df["riesgo"].value_counts(normalize=True) * 100)

print("\n📋 Primeros registros:")
print(df.head(10))

print("\n📊 Estadísticas por clase:")
for risk in ["Seguro", "Precaución", "Peligro"]:
    print(f"\n{risk}:")
    subset = df[df["riesgo"] == risk]
    print(f"  Temperatura: {subset['temperatura'].mean():.1f} ± {subset['temperatura'].std():.1f}")
    print(f"  Humedad: {subset['humedad'].mean():.1f} ± {subset['humedad'].std():.1f}")
    print(f"  Viento: {subset['viento'].mean():.1f} ± {subset['viento'].std():.1f}")
    print(f"  Visibilidad: {subset['visibilidad'].mean():.0f} ± {subset['visibilidad'].std():.0f}")

print("\n✅ Listo para entrenar con 'python scripts/train_model.py'")