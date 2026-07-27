"""
Servicio para obtener y parsear datos METAR y TAF
METAR: Meteorological Aerodrome Report
TAF: Terminal Aerodrome Forecast
"""
import httpx
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class METARTAFService:
    """
    Servicio para obtener reportes METAR y TAF de aeropuertos
    """
    
    # API de NOAA Aviation Weather Center
    METAR_URL = "https://aviationweather.gov/api/data/metar"
    TAF_URL = "https://aviationweather.gov/api/data/taf"
    
    def __init__(self):
        self.session = None
    
    async def get_metar_data(self, icao: str) -> Dict[str, Any]:
        """
        Obtiene reporte METAR de un aeropuerto
        
        Args:
            icao: Código ICAO del aeropuerto
            
        Returns:
            Datos METAR parseados
        """
        icao = icao.upper().strip()
        
        if len(icao) != 4:
            raise ValueError("Código ICAO debe tener 4 caracteres")
        
        try:
            # Parametros de la API de NOAA (aviationweather.gov).
            # 'taf' y 'date' NO son parametros validos de este endpoint y
            # provocaban un 400 Bad Request; se eliminaron. Con 'ids',
            # 'format' y 'hours' basta.
            params = {
                "ids": icao,
                "format": "raw",
                "hours": "2",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.METAR_URL,
                    params=params,
                    timeout=15.0
                )
                response.raise_for_status()
                raw_metar = response.text.strip()

            # NOAA puede devolver varias observaciones (una por hora), la
            # mas reciente primero. Se toma la primera linea: el METAR
            # vigente.
            if raw_metar:
                raw_metar = raw_metar.splitlines()[0].strip()
            
            if not raw_metar or "No METAR" in raw_metar:
                logger.warning(f"No hay METAR disponible para {icao}")
                raise ValueError(f"No hay METAR disponible para {icao}")
            
            # Parsear METAR
            parsed_data = self._parse_metar(raw_metar)
            
            logger.info(f"✅ METAR obtenido para {icao}")
            return {
                "icao": icao,
                "raw_metar": raw_metar,
                "parsed": parsed_data,
                "observation_time": parsed_data.get("observation_time"),
                "valid": True
            }
            
        except httpx.HTTPError as e:
            logger.error(f"Error HTTP obteniendo METAR: {e}")
            raise ValueError(f"No se pudo obtener METAR para {icao}")
        except Exception as e:
            logger.error(f"Error obteniendo METAR para {icao}: {e}")
            raise
    
    async def get_taf_data(self, icao: str) -> Dict[str, Any]:
        """
        Obtiene pronóstico TAF de un aeropuerto
        
        Args:
            icao: Código ICAO del aeropuerto
            
        Returns:
            Datos TAF parseados
        """
        icao = icao.upper().strip()
        
        if len(icao) != 4:
            raise ValueError("Código ICAO debe tener 4 caracteres")
        
        try:
            params = {
                "ids": icao,
                "format": "raw",
                "taf": "true",
                "hours": "0",
                "date": "0"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.TAF_URL,
                    params=params,
                    timeout=15.0
                )
                response.raise_for_status()
                raw_taf = response.text.strip()
            
            if not raw_taf or "No TAF" in raw_taf:
                logger.warning(f"No hay TAF disponible para {icao}")
                raise ValueError(f"No hay TAF disponible para {icao}")
            
            # Parsear TAF
            parsed_data = self._parse_taf(raw_taf)
            
            logger.info(f"✅ TAF obtenido para {icao}")
            return {
                "icao": icao,
                "raw_taf": raw_taf,
                "parsed": parsed_data,
                "issue_time": parsed_data.get("issue_time"),
                "valid_period": parsed_data.get("valid_period"),
                "valid": True
            }
            
        except httpx.HTTPError as e:
            logger.error(f"Error HTTP obteniendo TAF: {e}")
            raise ValueError(f"No se pudo obtener TAF para {icao}")
        except Exception as e:
            logger.error(f"Error obteniendo TAF para {icao}: {e}")
            raise
    
    def _parse_metar(self, metar_string: str) -> Dict[str, Any]:
        """
        Parsea un string METAR según estándares OACI
        
        Ejemplo METAR:
        METAR SKBO 011200Z 04008KT 9999 FEW020 SCT250 22/14 Q1018 NOSIG
        
        Componentes:
        - METAR: Tipo de reporte
        - SKBO: Código ICAO
        - 011200Z: Día y hora (01 del mes, 12:00 UTC)
        - 04008KT: Viento (040° a 8 nudos)
        - 9999: Visibilidad en metros
        - FEW020: Nubes dispersas a 2000 pies
        - 22/14: Temperatura/punto de rocío en °C
        - Q1018: QNH en hPa
        - NOSIG: Sin cambios significativos esperados
        """
        parsed = {
            "raw": metar_string,
            "type": "METAR"
        }
        
        parts = metar_string.split()
        
        try:
            # Índice para tracking
            i = 0
            
            # Tipo (METAR o SPECI)
            if parts[i] in ["METAR", "SPECI"]:
                parsed["report_type"] = parts[i]
                i += 1
            
            # ICAO
            if i < len(parts) and len(parts[i]) == 4:
                parsed["icao"] = parts[i]
                i += 1
            
            # Tiempo de observación (DDHHmmZ)
            if i < len(parts) and parts[i].endswith("Z"):
                time_str = parts[i]
                parsed["observation_time"] = self._parse_time(time_str)
                i += 1
            
            # AUTO (si está presente)
            if i < len(parts) and parts[i] == "AUTO":
                parsed["automated"] = True
                i += 1
            
            # Viento (dddssKT o dddssGggKT)
            if i < len(parts) and "KT" in parts[i]:
                wind_data = self._parse_wind(parts[i])
                parsed.update(wind_data)
                i += 1

            # Grupo de direccion de viento variable (dddVddd, p. ej.
            # '080V140'): informativo, no aporta al modelo. Se salta para
            # que no bloquee el parseo de la visibilidad.
            if i < len(parts) and re.fullmatch(r"\d{3}V\d{3}", parts[i]):
                i += 1

            # Visibilidad. Tres formas:
            #   - CAVOK: "Ceiling And Visibility OK" -> vis >= 10 km, sin
            #     nubes significativas ni fenomenos. Muy comun en METAR AUTO.
            #   - 4 digitos, con posible sufijo (9999, 9999NDV, 0500).
            #   - En millas terrestres (US): se ignora aqui, el modelo usa
            #     metros y las estaciones colombianas reportan en metros.
            if i < len(parts) and parts[i] == "CAVOK":
                parsed["visibility_m"] = 9999
                parsed["visibility_km"] = 9.999
                parsed["cavok"] = True
                i += 1
            elif i < len(parts) and re.match(r"^\d{4}(NDV)?$", parts[i]):
                vis = int(parts[i][:4])
                parsed["visibility_m"] = vis
                parsed["visibility_km"] = vis / 1000
                i += 1
            elif i < len(parts) and parts[i].isdigit():
                parsed["visibility_m"] = int(parts[i])
                parsed["visibility_km"] = int(parts[i]) / 1000
                i += 1

            # Nubes. El sufijo '///' (tipo no determinado por estacion
            # automatica) se tolera: _parse_clouds lee solo cobertura y
            # altura. Con CAVOK no hay grupo de nubes.
            clouds = []
            while i < len(parts) and parts[i][:3] in ["SKC", "CLR", "FEW", "SCT", "BKN", "OVC", "VV", "NSC", "NCD"]:
                cloud_data = self._parse_clouds(parts[i])
                if cloud_data:
                    clouds.append(cloud_data)
                i += 1
            if clouds:
                parsed["clouds"] = clouds
            
            # Temperatura/Punto de rocío (TT/DD o M02/M05)
            for j in range(i, len(parts)):
                if "/" in parts[j]:
                    temp_data = self._parse_temperature(parts[j])
                    parsed.update(temp_data)
                    break
            
            # QNH (Qxxxx o Axxxx)
            for j in range(i, len(parts)):
                if parts[j].startswith("Q") and parts[j][1:].isdigit():
                    parsed["qnh_hpa"] = int(parts[j][1:])
                    break
                elif parts[j].startswith("A") and parts[j][1:].isdigit():
                    # InHg a hPa
                    altimeter = int(parts[j][1:])
                    parsed["qnh_inhg"] = altimeter / 100
                    parsed["qnh_hpa"] = int(altimeter * 0.3386)
                    break
            
            # Fenómenos meteorológicos
            wx_phenomena = []
            for part in parts:
                if self._is_weather_phenomenon(part):
                    wx_phenomena.append(part)
            if wx_phenomena:
                parsed["weather_phenomena"] = wx_phenomena
            
        except Exception as e:
            logger.error(f"Error parseando METAR: {e}")
            parsed["parse_error"] = str(e)
        
        return parsed
    
    def _parse_taf(self, taf_string: str) -> Dict[str, Any]:
        """
        Parsea un string TAF
        
        Ejemplo TAF:
        TAF SKBO 011200Z 0112/0212 04008KT 9999 FEW020 SCT250
        TEMPO 0115/0118 6000 SHRA BKN015CB
        """
        parsed = {
            "raw": taf_string,
            "type": "TAF"
        }
        
        lines = taf_string.split("\n")
        main_line = lines[0].strip()
        parts = main_line.split()
        
        try:
            i = 0
            
            # TAF
            if parts[i] == "TAF":
                i += 1
            
            # AMD (Amended) si está presente
            if i < len(parts) and parts[i] == "AMD":
                parsed["amended"] = True
                i += 1
            
            # ICAO
            if i < len(parts):
                parsed["icao"] = parts[i]
                i += 1
            
            # Issue time (DDHHmmZ)
            if i < len(parts) and parts[i].endswith("Z"):
                parsed["issue_time"] = self._parse_time(parts[i])
                i += 1
            
            # Valid period (DDHH/DDHH)
            if i < len(parts) and "/" in parts[i]:
                valid_period = parts[i]
                parsed["valid_period"] = {
                    "raw": valid_period,
                    "from": valid_period.split("/")[0],
                    "to": valid_period.split("/")[1]
                }
                i += 1
            
            # Forecast periods
            forecast_periods = []
            current_period = {"type": "BASE"}
            
            # Parsear resto del TAF (similar a METAR)
            # Por simplicidad, guardamos el resto como raw
            parsed["forecast_raw"] = " ".join(parts[i:])
            
        except Exception as e:
            logger.error(f"Error parseando TAF: {e}")
            parsed["parse_error"] = str(e)
        
        return parsed
    
    def _parse_wind(self, wind_str: str) -> Dict[str, Any]:
        """Parsea componente de viento (04008KT o VRB05KT o 04008G15KT)"""
        wind_data = {}
        
        # Remover KT
        wind_str = wind_str.replace("KT", "").replace("MPS", "")
        
        if wind_str.startswith("VRB"):
            wind_data["wind_direction"] = "VRB"
            wind_data["wind_speed_kt"] = int(wind_str[3:5])
        elif "G" in wind_str:  # Gusts
            direction = wind_str[:3]
            speed_gust = wind_str[3:].split("G")
            wind_data["wind_direction"] = int(direction)
            wind_data["wind_speed_kt"] = int(speed_gust[0])
            wind_data["wind_gust_kt"] = int(speed_gust[1])
        else:
            wind_data["wind_direction"] = int(wind_str[:3])
            wind_data["wind_speed_kt"] = int(wind_str[3:5])
        
        # Convertir a km/h
        if "wind_speed_kt" in wind_data:
            wind_data["wind_speed_kmh"] = int(wind_data["wind_speed_kt"] * 1.852)
        if "wind_gust_kt" in wind_data:
            wind_data["wind_gust_kmh"] = int(wind_data["wind_gust_kt"] * 1.852)
        
        return wind_data
    
    def _parse_clouds(self, cloud_str: str) -> Optional[Dict[str, Any]]:
        """Parsea información de nubes (FEW020, BKN015CB)"""
        if cloud_str in ["SKC", "CLR"]:
            return {"coverage": "SKC", "description": "Sky Clear"}
        
        coverage_map = {
            "FEW": "Few",
            "SCT": "Scattered",
            "BKN": "Broken",
            "OVC": "Overcast",
            "VV": "Vertical Visibility"
        }
        
        coverage = cloud_str[:3]
        if coverage in coverage_map:
            cloud_data = {
                "coverage": coverage,
                "description": coverage_map[coverage]
            }
            
            # Altura (en cientos de pies)
            if len(cloud_str) >= 6:
                height_hundreds = cloud_str[3:6]
                if height_hundreds.isdigit():
                    cloud_data["height_ft"] = int(height_hundreds) * 100
            
            # Tipo de nube (CB = Cumulonimbus, TCU = Towering Cumulus)
            if "CB" in cloud_str:
                cloud_data["type"] = "Cumulonimbus"
            elif "TCU" in cloud_str:
                cloud_data["type"] = "Towering Cumulus"
            
            return cloud_data
        
        return None
    
    def _parse_temperature(self, temp_str: str) -> Dict[str, Any]:
        """Parsea temperatura y punto de rocío (22/14 o M02/M05)"""
        temp_data = {}
        
        if "/" in temp_str:
            temp_part, dewpoint_part = temp_str.split("/")
            
            # Temperatura
            if temp_part.startswith("M"):
                temp_data["temperature_c"] = -int(temp_part[1:])
            else:
                temp_data["temperature_c"] = int(temp_part)
            
            # Punto de rocío
            if dewpoint_part.startswith("M"):
                temp_data["dewpoint_c"] = -int(dewpoint_part[1:])
            else:
                temp_data["dewpoint_c"] = int(dewpoint_part)
        
        return temp_data
    
    def _parse_time(self, time_str: str) -> str:
        """Parsea tiempo de observación (DDHHmmZ)"""
        # Simplificado - en producción usar datetime
        return time_str
    
    def _is_weather_phenomenon(self, code: str) -> bool:
        """Verifica si es un código de fenómeno meteorológico"""
        wx_codes = [
            "RA", "SN", "DZ", "FG", "BR", "HZ", "TS", "SH",
            "FZ", "GR", "GS", "PL", "IC", "UP", "VA", "DS",
            "SS", "FC", "SQ", "+", "-", "VC", "MI", "PR", "BC", "DR", "BL"
        ]
        return any(code.startswith(wx) or code.endswith(wx) for wx in wx_codes)


# Instancia global del servicio
metar_taf_service = METARTAFService()


# ==================== HELPER FUNCTIONS ====================

async def get_metar_data(icao: str) -> Dict[str, Any]:
    """
    Helper function para obtener METAR
    
    Args:
        icao: Código ICAO del aeropuerto
        
    Returns:
        Datos METAR parseados
    """
    return await metar_taf_service.get_metar_data(icao)


async def get_taf_data(icao: str) -> Dict[str, Any]:
    """
    Helper function para obtener TAF
    
    Args:
        icao: Código ICAO del aeropuerto
        
    Returns:
        Datos TAF parseados
    """
    return await metar_taf_service.get_taf_data(icao)