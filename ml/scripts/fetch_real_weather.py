import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class WeatherDataCollector:
    """
    Recolector de datos meteorológicos de APIs reales para aviación
    """
    
    def __init__(self, openweather_key=None, checkwx_key=None):
        self.openweather_key = openweather_key
        self.checkwx_key = checkwx_key
        
    def get_openweather_data(self, lat, lon):
        """
        Obtiene datos de OpenWeatherMap
        API: https://openweathermap.org/api
        """
        if not self.openweather_key:
            print("⚠️ No hay API key de OpenWeatherMap")
            return None
            
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.openweather_key,
            'units': 'metric'  # Celsius
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'temperatura': data['main']['temp'],
                'temperatura_sensacion': data['main']['feels_like'],
                'presion': data['main']['pressure'],
                'humedad': data['main']['humidity'],
                'viento': data['wind']['speed'] * 3.6,  # m/s a km/h
                'direccion_viento': data['wind'].get('deg', 0),
                'rafagas': data['wind'].get('gust', 0) * 3.6 if 'gust' in data['wind'] else data['wind']['speed'] * 3.6,
                'visibilidad': data.get('visibility', 10000),
                'nubes': data['clouds']['all'],
                'descripcion_clima': data['weather'][0]['description'],
                'lluvia_1h': data.get('rain', {}).get('1h', 0),
                'timestamp': datetime.fromtimestamp(data['dt'])
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo datos de OpenWeather: {e}")
            return None
    
    def get_open_meteo_data(self, lat, lon):
        """
        Obtiene datos de Open-Meteo (gratis, sin límites)
        API: https://open-meteo.com/
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': [
                'temperature_2m',
                'relative_humidity_2m',
                'precipitation',
                'weather_code',
                'cloud_cover',
                'pressure_msl',
                'surface_pressure',
                'wind_speed_10m',
                'wind_direction_10m',
                'wind_gusts_10m'
            ],
            'hourly': [
                'visibility',
                'temperature_2m',
                'wind_speed_10m'
            ],
            'forecast_days': 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            current = data['current']
            
            # Convertir weather_code a descripción
            weather_codes = {
                0: 'despejado',
                1: 'mayormente_despejado',
                2: 'parcialmente_nublado',
                3: 'nublado',
                45: 'niebla',
                48: 'niebla_deposito_escarcha',
                51: 'llovizna_ligera',
                61: 'lluvia_ligera',
                63: 'lluvia_moderada',
                65: 'lluvia_intensa',
                71: 'nevada_ligera',
                95: 'tormenta'
            }
            
            return {
                'temperatura': current['temperature_2m'],
                'humedad': current['relative_humidity_2m'],
                'precipitacion': current['precipitation'],
                'codigo_clima': current['weather_code'],
                'descripcion_clima': weather_codes.get(current['weather_code'], 'desconocido'),
                'nubes': current['cloud_cover'],
                'presion_nivel_mar': current['pressure_msl'],
                'presion_superficie': current['surface_pressure'],
                'viento': current['wind_speed_10m'],
                'direccion_viento': current['wind_direction_10m'],
                'rafagas': current['wind_gusts_10m'],
                'visibilidad': data['hourly']['visibility'][0] if 'visibility' in data['hourly'] else 10000,
                'timestamp': datetime.fromisoformat(current['time'])
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo datos de Open-Meteo: {e}")
            return None
    
    def get_metar_data(self, icao_code):
        """
        Obtiene METAR (formato estándar de aviación)
        Requiere API key de CheckWX
        """
        if not self.checkwx_key:
            print("⚠️ No hay API key de CheckWX")
            return None
            
        url = f"https://api.checkwx.com/metar/{icao_code}/decoded"
        headers = {'X-API-Key': self.checkwx_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['results'] == 0:
                return None
                
            metar = data['data'][0]
            
            return {
                'icao': metar['icao'],
                'temperatura': metar.get('temperature', {}).get('celsius', None),
                'punto_rocio': metar.get('dewpoint', {}).get('celsius', None),
                'viento_velocidad': metar.get('wind', {}).get('speed_kts', 0) * 1.852,  # knots a km/h
                'viento_direccion': metar.get('wind', {}).get('degrees', 0),
                'visibilidad': metar.get('visibility', {}).get('meters', 10000),
                'presion_qnh': metar.get('barometer', {}).get('hpa', None),
                'techo_nubes': metar.get('ceiling', {}).get('feet', None),
                'condiciones': metar.get('conditions', []),
                'metar_raw': metar.get('raw_text', ''),
                'timestamp': datetime.fromisoformat(metar['observed'])
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo METAR: {e}")
            return None
    
    def calculate_aviation_features(self, weather_data, runway_heading=90):
        """
        Calcula features específicos de aviación
        """
        if not weather_data:
            return None
            
        features = weather_data.copy()
        
        # 1. Componente de viento cruzado (crítico)
        wind_dir = features.get('direccion_viento', 0)
        wind_speed = features.get('viento', 0)
        
        angle_diff = abs(wind_dir - runway_heading)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        crosswind = wind_speed * np.sin(np.radians(angle_diff))
        headwind = wind_speed * np.cos(np.radians(angle_diff))
        
        features['viento_cruzado'] = abs(crosswind)
        features['viento_frente'] = headwind
        
        # 2. Clasificación de riesgo de viento cruzado
        if abs(crosswind) > 35:
            features['riesgo_viento_cruzado'] = 'alto'
        elif abs(crosswind) > 20:
            features['riesgo_viento_cruzado'] = 'medio'
        else:
            features['riesgo_viento_cruzado'] = 'bajo'
        
        # 3. Condiciones IMC/VMC (Instrumental/Visual)
        vis = features.get('visibilidad', 10000)
        clouds = features.get('techo_nubes', 10000)
        
        # Mínimos VMC: visibilidad > 5km y techo > 1000ft
        if vis < 5000 or (clouds and clouds < 1000):
            features['condiciones_vuelo'] = 'IMC'  # Instrumental
        else:
            features['condiciones_vuelo'] = 'VMC'  # Visual
        
        # 4. Índice de densidad del aire (afecta performance)
        temp = features.get('temperatura', 15)
        pressure = features.get('presion', 1013)
        
        # Densidad relativa (aprox)
        density_ratio = (pressure / 1013.25) * (288.15 / (temp + 273.15))
        features['densidad_aire'] = density_ratio
        
        # 5. Riesgo de formación de hielo
        temp_rocio = features.get('punto_rocio', temp - 5)
        if 0 <= temp <= 10 and (temp - temp_rocio) < 3:
            features['riesgo_hielo'] = True
        else:
            features['riesgo_hielo'] = False
        
        # 6. Índice de riesgo compuesto (0-100)
        risk_score = 0
        
        # Visibilidad
        if vis < 1000:
            risk_score += 30
        elif vis < 3000:
            risk_score += 20
        elif vis < 5000:
            risk_score += 10
        
        # Viento cruzado
        if abs(crosswind) > 35:
            risk_score += 30
        elif abs(crosswind) > 20:
            risk_score += 15
        
        # Ráfagas
        gusts = features.get('rafagas', wind_speed)
        if gusts > 50:
            risk_score += 20
        elif gusts > 35:
            risk_score += 10
        
        # Temperatura extrema
        if temp < 0 or temp > 35:
            risk_score += 10
        
        # Precipitación
        rain = features.get('lluvia_1h', 0) or features.get('precipitacion', 0)
        if rain > 10:
            risk_score += 15
        elif rain > 5:
            risk_score += 8
        
        features['indice_riesgo'] = min(risk_score, 100)
        
        # 7. Clasificación final
        if risk_score >= 70 or features['condiciones_vuelo'] == 'IMC':
            features['riesgo'] = 'alto'
        elif risk_score >= 40:
            features['riesgo'] = 'medio'
        else:
            features['riesgo'] = 'bajo'
        
        return features
    
    def collect_historical_data(self, locations, days=30):
        """
        Recolecta datos históricos de múltiples ubicaciones
        """
        all_data = []
        
        for location in locations:
            lat, lon, name = location['lat'], location['lon'], location['name']
            print(f"\n📍 Recolectando datos de {name}...")
            
            for day in range(days):
                date = datetime.now() - timedelta(days=day)
                
                # Open-Meteo (gratis)
                data = self.get_open_meteo_data(lat, lon)
                
                if data:
                    # Calcular features de aviación
                    aviation_data = self.calculate_aviation_features(
                        data, 
                        runway_heading=location.get('runway_heading', 90)
                    )
                    
                    aviation_data['ubicacion'] = name
                    aviation_data['latitud'] = lat
                    aviation_data['longitud'] = lon
                    
                    all_data.append(aviation_data)
                
                # Respetar rate limits
                time.sleep(1)
                
                if (day + 1) % 10 == 0:
                    print(f"  ✓ {day + 1}/{days} días procesados")
        
        return pd.DataFrame(all_data)


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    
    # Configurar API keys (obtenerlas gratis en las páginas)
    OPENWEATHER_KEY = "a45d492668dceb132d0d67106b718810"  # https://openweathermap.org/api
    CHECKWX_KEY = None  # https://www.checkwx.com/ (opcional)
    
    # Crear colector
    collector = WeatherDataCollector(
        openweather_key=OPENWEATHER_KEY,
        checkwx_key=CHECKWX_KEY
    )
    
    # Definir aeropuertos/ubicaciones
    locations = [
        {
            'name': 'Bogotá (El Dorado)',
            'lat': 4.7016,
            'lon': -74.1469,
            'icao': 'SKBO',
            'runway_heading': 131  # Pista 13R
        },
        {
            'name': 'Medellín (Olaya Herrera)',
            'lat': 6.2204,
            'lon': -75.5906,
            'icao': 'SKMD',
            'runway_heading': 20
        },
        {
            'name': 'Cali (Alfonso Bonilla)',
            'lat': 3.5432,
            'lon': -76.3816,
            'icao': 'SKCL',
            'runway_heading': 19
        }
    ]
    
    # OPCIÓN 1: Obtener datos en tiempo real
    print("=" * 60)
    print("🌦️  DATOS METEOROLÓGICOS EN TIEMPO REAL")
    print("=" * 60)
    
    for location in locations:
        print(f"\n📍 {location['name']}")
        
        # Método 1: Open-Meteo (gratis, sin límites)
        data = collector.get_open_meteo_data(location['lat'], location['lon'])
        
        if data:
            # Calcular features de aviación
            aviation_data = collector.calculate_aviation_features(
                data,
                runway_heading=location['runway_heading']
            )
            
            print(f"  🌡️  Temperatura: {aviation_data['temperatura']:.1f}°C")
            print(f"  💨 Viento: {aviation_data['viento']:.1f} km/h ({aviation_data['direccion_viento']:.0f}°)")
            print(f"  ✈️  Viento Cruzado: {aviation_data['viento_cruzado']:.1f} km/h")
            print(f"  👁️  Visibilidad: {aviation_data['visibilidad']:.0f} m")
            print(f"  ☁️  Nubes: {aviation_data['nubes']}%")
            print(f"  📊 Condiciones: {aviation_data['condiciones_vuelo']}")
            print(f"  ⚠️  Riesgo: {aviation_data['riesgo'].upper()}")
            print(f"  📈 Índice de Riesgo: {aviation_data['indice_riesgo']}/100")
    
    # OPCIÓN 2: Recolectar datos históricos para entrenamiento
    print("\n" + "=" * 60)
    print("📦 RECOLECTANDO DATOS HISTÓRICOS (30 días)")
    print("=" * 60)
    
    # Descomentar para recolectar datos históricos
    # df_historical = collector.collect_historical_data(locations, days=30)
    # df_historical.to_csv('data/dataset/weather_real_data.csv', index=False)
    # print(f"\n✅ {len(df_historical)} registros guardados en 'weather_real_data.csv'")
    
    print("\n💡 Para usar:")
    print("  1. Obtén API keys gratis en:")
    print("     - https://openweathermap.org/api")
    print("     - https://open-meteo.com (no requiere key)")
    print("  2. Reemplaza 'tu_api_key_aqui' con tu key")
    print("  3. Ejecuta el script para recolectar datos reales")
    print("  4. Usa los datos para reentrenar el modelo")