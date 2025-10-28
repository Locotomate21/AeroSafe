# scripts/analizar_datos.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv("data/dataset/weather_risk.csv")

print("=" * 60)
print("📊 ANÁLISIS DETALLADO DEL DATASET")
print("=" * 60)

# 1. Verificar correlación entre features y riesgo
print("\n1️⃣ CORRELACIÓN ENTRE VARIABLES Y RIESGO")
print("-" * 60)

# Codificar riesgo numéricamente para correlación
risk_map = {'Seguro': 0, 'Precaución': 1, 'Peligro': 2}
df['riesgo_num'] = df['riesgo'].map(risk_map)

correlations = df[['temperatura', 'humedad', 'viento', 'visibilidad', 'riesgo_num']].corr()['riesgo_num'].sort_values(ascending=False)
print(correlations)

# 2. Analizar separabilidad de clases
print("\n2️⃣ ESTADÍSTICAS POR CLASE")
print("-" * 60)
for col in ['temperatura', 'humedad', 'viento', 'visibilidad']:
    print(f"\n{col.upper()}:")
    print(df.groupby('riesgo')[col].describe()[['mean', 'std', 'min', 'max']])

# 3. Buscar solapamiento entre clases
print("\n3️⃣ ANÁLISIS DE SOLAPAMIENTO")
print("-" * 60)

for risk in ['Seguro', 'Precaución', 'Peligro']:
    subset = df[df['riesgo'] == risk]
    print(f"\n{risk}:")
    print(f"  Temperatura: {subset['temperatura'].min():.1f} - {subset['temperatura'].max():.1f}")
    print(f"  Humedad: {subset['humedad'].min():.0f} - {subset['humedad'].max():.0f}")
    print(f"  Viento: {subset['viento'].min():.1f} - {subset['viento'].max():.1f}")
    print(f"  Visibilidad: {subset['visibilidad'].min():.0f} - {subset['visibilidad'].max():.0f}")

# 4. Verificar lógica de clasificación
print("\n4️⃣ VERIFICACIÓN DE LÓGICA DE CLASIFICACIÓN")
print("-" * 60)

# Mostrar ejemplos de cada clase
print("\nEjemplos de SEGURO:")
print(df[df['riesgo'] == 'Seguro'][['temperatura', 'humedad', 'viento', 'visibilidad']].head(5))

print("\nEjemplos de PRECAUCIÓN:")
print(df[df['riesgo'] == 'Precaución'][['temperatura', 'humedad', 'viento', 'visibilidad']].head(5))

print("\nEjemplos de PELIGRO:")
print(df[df['riesgo'] == 'Peligro'][['temperatura', 'humedad', 'viento', 'visibilidad']].head(5))

# 5. Buscar inconsistencias
print("\n5️⃣ BÚSQUEDA DE INCONSISTENCIAS")
print("-" * 60)

# Agrupar por valores similares y ver si tienen diferentes etiquetas
df['temp_bin'] = pd.cut(df['temperatura'], bins=10, labels=False)
df['hum_bin'] = pd.cut(df['humedad'], bins=10, labels=False)
df['viento_bin'] = pd.cut(df['viento'], bins=10, labels=False)
df['vis_bin'] = pd.cut(df['visibilidad'], bins=10, labels=False)

grouped = df.groupby(['temp_bin', 'hum_bin', 'viento_bin', 'vis_bin'])['riesgo'].apply(lambda x: x.nunique())
inconsistent = grouped[grouped > 1]

print(f"\nNúmero de grupos con condiciones similares pero diferentes etiquetas: {len(inconsistent)}")
print(f"Esto representa {len(inconsistent) / len(grouped) * 100:.1f}% de los grupos")

# 6. Revisar reglas obvias
print("\n6️⃣ VERIFICACIÓN DE REGLAS OBVIAS")
print("-" * 60)

# Condiciones que deberían ser peligrosas
dangerous_conditions = df[
    ((df['viento'] > 40) | (df['visibilidad'] < 2000) | (df['humedad'] > 95))
]
print(f"\nCasos con condiciones obviamente peligrosas: {len(dangerous_conditions)}")
print("Distribución de riesgo en estos casos:")
print(dangerous_conditions['riesgo'].value_counts())

# Condiciones que deberían ser seguras
safe_conditions = df[
    (df['temperatura'].between(15, 30)) & 
    (df['humedad'].between(40, 70)) & 
    (df['viento'] < 15) & 
    (df['visibilidad'] > 8000)
]
print(f"\nCasos con condiciones obviamente seguras: {len(safe_conditions)}")
print("Distribución de riesgo en estos casos:")
print(safe_conditions['riesgo'].value_counts())

# 7. Proponer nuevas reglas
print("\n7️⃣ PROPUESTA DE REGLAS MEJORADAS")
print("-" * 60)
print("""
Basado en el análisis, considera estas reglas:

PELIGRO:
- Viento > 40 km/h
- Visibilidad < 2000 m
- Humedad > 95% Y (Viento > 30 O Temperatura < 5)
- Temperatura < 0°C O > 38°C

PRECAUCIÓN:
- Viento entre 25-40 km/h
- Visibilidad entre 2000-5000 m
- Humedad > 85%
- Temperatura entre 0-10°C O 33-38°C

SEGURO:
- Todo lo demás que no cumpla las anteriores
""")

print("\n✅ Análisis completado")
print("\n💡 RECOMENDACIONES:")
print("1. Revisa el archivo 'generate_dataset.py' y ajusta las reglas de clasificación")
print("2. Las clases parecen estar solapadas - considera simplificar a 2 clases (Seguro/Peligroso)")
print("3. O regenera el dataset con reglas más estrictas y menos ambiguas")