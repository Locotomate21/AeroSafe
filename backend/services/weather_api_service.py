# backend/services/weather_api_service.py
import httpx
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from core.config import settings

logger = logging.getLogger(__name__)

class WeatherAPIService:
    """Servicio para obtener datos meteorológicos reales"""
    
    def __init__(self):
        self.api_key = settings.WEATHER_API_KEY or settings.OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.base_url_history = "https://history.openweathermap.org/data/2.5"
        
    async def get_current_weather(self, city: str = None, lat: float = None, lon: float = None) -> Dict:
        """
        Obtiene clima actual de una ciudad o coordenadas
        
        Args:
            city: Nombre de la ciudad (ej: "Bogotá,CO")
            lat: Latitud
            lon: Longitud
        """
        try:
            url = f"{self.base_url}/weather"
            
            params = {
                "appid": self.api_key,
                "units": "metric"  # Celsius
            }
            
            if city:
                params["q"] = city
            elif lat and lon:
                params["lat"] = lat
                params["lon"] = lon
            else:
                raise ValueError("Debe proporcionar city o (lat, lon)")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
            
            # Extraer datos relevantes
            weather_data = self._parse_current_weather(data)
            logger.info(f"Datos obtenidos para {city or f'{lat},{lon}'}")
            
            return weather_data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Error obteniendo clima: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            raise
    
    async def get_forecast(self, city: str, days: int = 5) -> Dict:
        """
        Obtiene pronóstico de 5 días (intervalos de 3 horas)
        """
        try:
            url = f"{self.base_url}/forecast"
            
            params = {
                "q": city,
                "appid": self.api_key,
                "units": "metric",
                "cnt": min(days * 8, 40)  # Máx 40 intervalos (5 días)
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
            
            forecast_list = []
            for item in data.get("list", []):
                forecast_list.append(self._parse_forecast_item(item))
            
            return {
                "city": data["city"]["name"],
                "country": data["city"]["country"],
                "forecast": forecast_list
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo pronóstico: {str(e)}")
            raise
    
    async def get_airport_weather(self, icao_code: str) -> Dict:
        """
        Obtiene clima específico de aeropuerto usando código ICAO
        Ej: SKBO (Bogotá), SKCL (Cali), SKMR (Medellín)
        """
        # Mapeo de códigos ICAO a coordenadas principales de Colombia
        airports = {
            "SKBO": {"name": "El Dorado (Bogotá)", "lat": 4.7016, "lon": -74.1469},
            "SKCL": {"name": "Alfonso Bonilla (Cali)", "lat": 3.5432, "lon": -76.3816},
            "SKMR": {"name": "Olaya Herrera (Medellín)", "lat": 6.2205, "lon": -75.5909},
            "SKRG": {"name": "José María Córdova (Rionegro)", "lat": 6.1645, "lon": -75.4233},
            "SKCG": {"name": "Rafael Núñez (Cartagena)", "lat": 10.4424, "lon": -75.5130},
            "SKBQ": {"name": "Palonegro (Bucaramanga)", "lat": 7.1265, "lon": -73.1848},
            "SKPE": {"name": "Matecaña (Pereira)", "lat": 4.8127, "lon": -75.7395},
        }
        
        if icao_code.upper() not in airports:
            raise ValueError(f"Código ICAO {icao_code} no soportado. Disponibles: {list(airports.keys())}")
        
        airport = airports[icao_code.upper()]
        weather = await self.get_current_weather(lat=airport["lat"], lon=airport["lon"])
        weather["airport_name"] = airport["name"]
        weather["icao_code"] = icao_code.upper()
        
        return weather
    
    def _parse_current_weather(self, data: Dict) -> Dict:
        """Convierte respuesta de API a formato estandarizado"""
        return {
            "ciudad": data.get("name", ""),
            "pais": data.get("sys", {}).get("country", ""),
            "timestamp": datetime.fromtimestamp(data.get("dt", 0)).isoformat(),
            
            # Variables principales
            "temperatura": data.get("main", {}).get("temp", 0),
            "temperatura_sensacion": data.get("main", {}).get("feels_like", 0),
            "temp_min": data.get("main", {}).get("temp_min", 0),
            "temp_max": data.get("main", {}).get("temp_max", 0),
            
            "humedad": data.get("main", {}).get("humidity", 0),
            "presion": data.get("main", {}).get("pressure", 0),
            "presion_nivel_mar": data.get("main", {}).get("sea_level", 0),
            
            "viento": data.get("wind", {}).get("speed", 0) * 3.6,  # m/s a km/h
            "viento_direccion": data.get("wind", {}).get("deg", 0),
            "viento_rafagas": data.get("wind", {}).get("gust", 0) * 3.6,
            
            "visibilidad": data.get("visibility", 0),
            "nubes": data.get("clouds", {}).get("all", 0),
            
            # Condiciones
            "descripcion": data.get("weather", [{}])[0].get("description", ""),
            "condicion_principal": data.get("weather", [{}])[0].get("main", ""),
            
            # Lluvia/Nieve (última hora)
            "lluvia_1h": data.get("rain", {}).get("1h", 0),
            "nieve_1h": data.get("snow", {}).get("1h", 0),
            
            # Coordenadas
            "latitud": data.get("coord", {}).get("lat", 0),
            "longitud": data.get("coord", {}).get("lon", 0),
        }
    
    def _parse_forecast_item(self, item: Dict) -> Dict:
        """Parsea item de pronóstico"""
        return {
            "timestamp": datetime.fromtimestamp(item.get("dt", 0)).isoformat(),
            "temperatura": item.get("main", {}).get("temp", 0),
            "humedad": item.get("main", {}).get("humidity", 0),
            "viento": item.get("wind", {}).get("speed", 0) * 3.6,
            "visibilidad": item.get("visibility", 0),
            "descripcion": item.get("weather", [{}])[0].get("description", ""),
            "probabilidad_precipitacion": item.get("pop", 0) * 100,
        }

# Singleton instance
weather_api_service = WeatherAPIService()