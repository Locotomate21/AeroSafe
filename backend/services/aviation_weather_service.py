"""
Servicio para obtener datos meteorológicos de OpenWeather API
"""
import httpx
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from core.config import settings

logger = logging.getLogger(__name__)


class WeatherService:
    """
    Servicio para interactuar con OpenWeather API
    """
    
    def __init__(self):
        self.api_key = settings.OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        if not self.api_key:
            logger.warning("⚠️ OPENWEATHER_API_KEY no configurada - algunas funciones no estarán disponibles")
    
    async def get_current_weather(self, city: str) -> Dict[str, Any]:
        """
        Obtiene clima actual de una ciudad
        
        Args:
            city: Ciudad en formato "Ciudad,Código_País" (ej: "Bogotá,CO")
            
        Returns:
            Datos meteorológicos actuales
            
        Raises:
            ValueError: Si el formato de ciudad es inválido
            httpx.HTTPError: Si hay error en la API
        """
        if not self.api_key:
            raise ValueError("API key de OpenWeather no configurada")
        
        # Validar formato
        if "," not in city:
            raise ValueError("Ciudad debe estar en formato 'Ciudad,País' (ej: 'Bogotá,CO')")
        
        city_name, country_code = city.split(",", 1)
        city_name = city_name.strip()
        country_code = country_code.strip().upper()
        
        # Construir query
        query = f"{city_name},{country_code}"
        
        url = f"{self.base_url}/weather"
        params = {
            "q": query,
            "appid": self.api_key,
            "units": "metric",  # Celsius
            "lang": "es"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
            
            # Transformar respuesta a formato estándar
            weather_data = self._parse_weather_response(data)
            logger.info(f"✅ Clima obtenido para {city}")
            
            return weather_data
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error(f"Ciudad no encontrada: {city}")
                raise ValueError(f"Ciudad '{city}' no encontrada")
            else:
                logger.error(f"Error HTTP obteniendo clima: {e}")
                raise
        except httpx.HTTPError as e:
            logger.error(f"Error de red obteniendo clima: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado obteniendo clima: {e}")
            raise
    
    async def get_weather_by_coords(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Obtiene clima por coordenadas geográficas
        
        Args:
            lat: Latitud (-90 a 90)
            lon: Longitud (-180 a 180)
            
        Returns:
            Datos meteorológicos
        """
        if not self.api_key:
            raise ValueError("API key de OpenWeather no configurada")
        
        # Validar coordenadas
        if not -90 <= lat <= 90:
            raise ValueError("Latitud debe estar entre -90 y 90")
        if not -180 <= lon <= 180:
            raise ValueError("Longitud debe estar entre -180 y 180")
        
        url = f"{self.base_url}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "lang": "es"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
            
            weather_data = self._parse_weather_response(data)
            logger.info(f"✅ Clima obtenido para coords ({lat}, {lon})")
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error obteniendo clima por coordenadas: {e}")
            raise
    
    async def get_forecast_data(self, city: str, days: int = 3) -> Dict[str, Any]:
        """
        Obtiene pronóstico meteorológico
        
        Args:
            city: Ciudad en formato "Ciudad,Código_País"
            days: Días de pronóstico (1-5)
            
        Returns:
            Pronóstico meteorológico
        """
        if not self.api_key:
            raise ValueError("API key de OpenWeather no configurada")
        
        if not 1 <= days <= 5:
            raise ValueError("Días debe estar entre 1 y 5")
        
        # Validar formato ciudad
        if "," not in city:
            raise ValueError("Ciudad debe estar en formato 'Ciudad,País'")
        
        city_name, country_code = city.split(",", 1)
        query = f"{city_name.strip()},{country_code.strip().upper()}"
        
        url = f"{self.base_url}/forecast"
        params = {
            "q": query,
            "appid": self.api_key,
            "units": "metric",
            "lang": "es",
            "cnt": days * 8  # 8 mediciones por día (cada 3h)
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
            
            forecast_data = self._parse_forecast_response(data, days)
            logger.info(f"✅ Pronóstico obtenido para {city} ({days} días)")
            
            return forecast_data
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Ciudad '{city}' no encontrada")
            raise
        except Exception as e:
            logger.error(f"Error obteniendo pronóstico: {e}")
            raise
    
    def _parse_weather_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parsea respuesta de OpenWeather a formato estándar
        
        Args:
            data: Response JSON de OpenWeather
            
        Returns:
            Datos meteorológicos en formato estándar
        """
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        clouds = data.get("clouds", {})
        rain = data.get("rain", {})
        
        return {
            "temperatura": main.get("temp", 0),
            "temp_min": main.get("temp_min"),
            "temp_max": main.get("temp_max"),
            "sensacion_termica": main.get("feels_like"),
            "humedad": main.get("humidity", 0),
            "presion": main.get("pressure", 1013),
            "viento": wind.get("speed", 0) * 3.6,  # m/s a km/h
            "direccion_viento": wind.get("deg"),
            "rafagas_viento": wind.get("gust", 0) * 3.6 if wind.get("gust") else None,
            "visibilidad": data.get("visibility", 10000),
            "nubes": clouds.get("all", 0),
            "precipitacion": rain.get("1h", 0) if rain else 0,
            "condicion": weather.get("main", "Clear"),
            "descripcion": weather.get("description", ""),
            "icono": weather.get("icon", ""),
            "ciudad": data.get("name", ""),
            "pais": data.get("sys", {}).get("country", ""),
            "coordenadas": {
                "lat": data.get("coord", {}).get("lat"),
                "lon": data.get("coord", {}).get("lon")
            },
            "timestamp": datetime.utcfromtimestamp(data.get("dt", 0)).isoformat(),
            "amanecer": datetime.utcfromtimestamp(
                data.get("sys", {}).get("sunrise", 0)
            ).isoformat() if data.get("sys", {}).get("sunrise") else None,
            "atardecer": datetime.utcfromtimestamp(
                data.get("sys", {}).get("sunset", 0)
            ).isoformat() if data.get("sys", {}).get("sunset") else None,
        }
    
    def _parse_forecast_response(self, data: Dict[str, Any], days: int) -> Dict[str, Any]:
        """
        Parsea respuesta de pronóstico
        
        Args:
            data: Response JSON de OpenWeather
            days: Días solicitados
            
        Returns:
            Pronóstico en formato estándar
        """
        forecasts = []
        
        for item in data.get("list", []):
            forecast_item = {
                "fecha": datetime.utcfromtimestamp(item.get("dt", 0)).isoformat(),
                "temperatura": item.get("main", {}).get("temp", 0),
                "temp_min": item.get("main", {}).get("temp_min"),
                "temp_max": item.get("main", {}).get("temp_max"),
                "humedad": item.get("main", {}).get("humidity", 0),
                "presion": item.get("main", {}).get("pressure", 1013),
                "viento": item.get("wind", {}).get("speed", 0) * 3.6,
                "direccion_viento": item.get("wind", {}).get("deg"),
                "visibilidad": item.get("visibility", 10000),
                "nubes": item.get("clouds", {}).get("all", 0),
                "precipitacion": item.get("rain", {}).get("3h", 0) if item.get("rain") else 0,
                "condicion": item.get("weather", [{}])[0].get("main", "Clear"),
                "descripcion": item.get("weather", [{}])[0].get("description", ""),
                "probabilidad_precipitacion": item.get("pop", 0) * 100,
            }
            forecasts.append(forecast_item)
        
        return {
            "location": f"{data.get('city', {}).get('name', '')},{data.get('city', {}).get('country', '')}",
            "forecast_days": days,
            "total_forecasts": len(forecasts),
            "forecasts": forecasts,
            "coordenadas": {
                "lat": data.get("city", {}).get("coord", {}).get("lat"),
                "lon": data.get("city", {}).get("coord", {}).get("lon")
            }
        }


# Instancia global del servicio
weather_service = WeatherService()


# ==================== HELPER FUNCTIONS ====================

async def get_weather_data(city: str) -> Dict[str, Any]:
    """
    Helper function para obtener clima actual
    
    Args:
        city: Ciudad en formato "Ciudad,País"
        
    Returns:
        Datos meteorológicos
    """
    return await weather_service.get_current_weather(city)


async def get_weather_by_coords(lat: float, lon: float) -> Dict[str, Any]:
    """
    Helper function para obtener clima por coordenadas
    
    Args:
        lat: Latitud
        lon: Longitud
        
    Returns:
        Datos meteorológicos
    """
    return await weather_service.get_weather_by_coords(lat, lon)


async def get_forecast_data(city: str, days: int = 3) -> Dict[str, Any]:
    """
    Helper function para obtener pronóstico
    
    Args:
        city: Ciudad
        days: Días de pronóstico
        
    Returns:
        Pronóstico meteorológico
    """
    return await weather_service.get_forecast_data(city, days)