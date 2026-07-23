from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

# ==================== RISK SCHEMAS ====================

class RiskRequest(BaseModel):
    """Request para predicción de riesgo"""
    temperatura: float = Field(..., description="Temperatura en °C")
    humedad: float = Field(..., ge=0, le=100, description="Humedad relativa %")
    viento: float = Field(..., ge=0, description="Velocidad del viento en km/h")
    visibilidad: float = Field(..., ge=0, description="Visibilidad en metros")
    presion: Optional[float] = Field(1013, description="Presión atmosférica en hPa")
    condicion: Optional[str] = Field("Unknown", description="Condición meteorológica")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperatura": 15.5,
                "humedad": 85,
                "viento": 35,
                "visibilidad": 2500,
                "presion": 1013,
                "condicion": "Tormenta"
            }
        }


class RiskResponse(BaseModel):
    """Response con predicción de riesgo"""
    riesgo: str = Field(..., description="Nivel de riesgo: BAJO, MODERADO, ALTO, CRÍTICO")
    confianza: float = Field(..., ge=0, le=1, description="Confianza de la predicción (0-1)")
    probabilidades: Dict[str, float] = Field(..., description="Probabilidades por clase")
    factores_riesgo: Optional[List[str]] = Field(default=[], description="Factores que contribuyen al riesgo")
    recomendaciones: Optional[List[str]] = Field(default=[], description="Recomendaciones operacionales")
    datos_clima: Dict[str, Any] = Field(..., description="Datos meteorológicos usados")
    timestamp: Optional[str] = Field(default=None, description="Timestamp de la predicción")
    
    class Config:
        json_schema_extra = {
            "example": {
                "riesgo": "ALTO",
                "confianza": 0.92,
                "probabilidades": {
                    "BAJO": 0.02,
                    "MODERADO": 0.06,
                    "ALTO": 0.92,
                    "CRÍTICO": 0.00
                },
                "factores_riesgo": [
                    "Viento fuerte (35 km/h)",
                    "Visibilidad reducida (2500m)"
                ],
                "recomendaciones": [
                    "Implementar procedimientos IFR",
                    "Monitorear evolución del viento"
                ],
                "datos_clima": {
                    "temperatura": 15.5,
                    "humedad": 85,
                    "viento": 35,
                    "visibilidad": 2500
                },
                "timestamp": "2024-02-01T12:00:00Z"
            }
        }


# ==================== WEATHER SCHEMAS ====================

class WeatherResponse(BaseModel):
    """Response con datos meteorológicos"""
    temperatura: float = Field(..., description="Temperatura en °C")
    humedad: float = Field(..., description="Humedad relativa %")
    viento: float = Field(..., description="Velocidad del viento en km/h")
    direccion_viento: Optional[int] = Field(None, description="Dirección del viento en grados")
    visibilidad: float = Field(..., description="Visibilidad en metros")
    presion: float = Field(..., description="Presión atmosférica en hPa")
    condicion: str = Field(..., description="Condición meteorológica")
    descripcion: Optional[str] = Field(None, description="Descripción detallada")
    nubes: Optional[int] = Field(None, description="Cobertura de nubes %")
    precipitacion: Optional[float] = Field(None, description="Precipitación en mm")
    timestamp: str = Field(..., description="Timestamp de la observación")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperatura": 18.5,
                "humedad": 70,
                "viento": 12,
                "direccion_viento": 45,
                "visibilidad": 9000,
                "presion": 1013,
                "condicion": "Nublado",
                "descripcion": "Nubes dispersas",
                "nubes": 50,
                "precipitacion": 0,
                "timestamp": "2024-02-01T12:00:00Z"
            }
        }


class AirportWeatherResponse(WeatherResponse):
    """Response con datos meteorológicos de aeropuerto"""
    icao: str = Field(..., description="Código ICAO del aeropuerto")
    airport_name: str = Field(..., description="Nombre del aeropuerto")
    elevation: Optional[int] = Field(None, description="Elevación del aeropuerto en pies")
    location: Optional[Dict[str, float]] = Field(None, description="Coordenadas lat/lon")
    
    class Config:
        json_schema_extra = {
            "example": {
                "icao": "SKBO",
                "airport_name": "El Dorado International Airport",
                "elevation": 8361,
                "location": {"lat": 4.7016, "lon": -74.1469},
                "temperatura": 18.5,
                "humedad": 70,
                "viento": 12,
                "direccion_viento": 40,
                "visibilidad": 9000,
                "presion": 756,
                "condicion": "Parcialmente nublado",
                "descripcion": "Nubes dispersas a 2000 pies",
                "nubes": 30,
                "precipitacion": 0,
                "timestamp": "2024-02-01T12:00:00Z"
            }
        }


class METARResponse(BaseModel):
    """Response con datos METAR"""
    icao: str = Field(..., description="Código ICAO")
    raw_metar: str = Field(..., description="METAR raw")
    observation_time: str = Field(..., description="Tiempo de observación")
    temperatura: float
    dewpoint: Optional[float] = None
    viento_velocidad: Optional[float] = None
    viento_direccion: Optional[int] = None
    visibilidad: float
    qnh: Optional[float] = None
    nubes: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "icao": "SKBO",
                "raw_metar": "METAR SKBO 011200Z 04008KT 9999 FEW020 SCT250 22/14 Q1018",
                "observation_time": "2024-02-01T12:00:00Z",
                "temperatura": 22,
                "dewpoint": 14,
                "viento_velocidad": 8,
                "viento_direccion": 40,
                "visibilidad": 9999,
                "qnh": 1018,
                "nubes": [
                    {"type": "FEW", "height": 2000},
                    {"type": "SCT", "height": 25000}
                ]
            }
        }


class TAFResponse(BaseModel):
    """Response con pronóstico TAF"""
    icao: str
    raw_taf: str
    issue_time: str
    valid_period: Dict[str, str]
    forecast_periods: List[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "icao": "SKBO",
                "raw_taf": "TAF SKBO 011200Z 0112/0212 04008KT 9999 FEW020 SCT250",
                "issue_time": "2024-02-01T12:00:00Z",
                "valid_period": {
                    "from": "2024-02-01T12:00:00Z",
                    "to": "2024-02-02T12:00:00Z"
                },
                "forecast_periods": [
                    {
                        "type": "FM",
                        "time": "2024-02-01T18:00:00Z",
                        "wind": {"speed": 12, "direction": 45},
                        "visibility": 9000,
                        "weather": "SCT020"
                    }
                ]
            }
        }


class ForecastResponse(BaseModel):
    """Response con pronóstico meteorológico"""
    location: str
    forecast_days: int
    forecasts: List[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": "Bogotá,CO",
                "forecast_days": 3,
                "forecasts": [
                    {
                        "date": "2024-02-01",
                        "temp_max": 22,
                        "temp_min": 12,
                        "humidity": 70,
                        "wind": 10,
                        "condition": "Parcialmente nublado"
                    }
                ]
            }
        }


# ==================== DASHBOARD SCHEMAS ====================

class DashboardStatus(BaseModel):
    """Estado general del sistema"""
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]
    metrics: Dict[str, Any]


class AirportStatus(BaseModel):
    """Estado completo de un aeropuerto"""
    airport: Dict[str, Any]
    current_weather: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    operational_impact: Dict[str, Any]
    recommendations: List[str]
    last_updated: str


# ==================== GENERIC SCHEMAS ====================

class HealthResponse(BaseModel):
    """Response de health check"""
    status: str
    service: str
    version: Optional[str] = None
    timestamp: Optional[str] = None


class ErrorResponse(BaseModel):
    """Response de error estándar"""
    detail: str
    error_code: Optional[str] = None
    timestamp: Optional[str] = None


class SuccessResponse(BaseModel):
    """Response genérico de éxito"""
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None