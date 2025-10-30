from pydantic import BaseModel, Field
from typing import Optional, Dict

class RiskRequest(BaseModel):
    temperatura: float = Field(..., description="Temperatura en °C")
    humedad: float = Field(..., ge=0, le=100, description="Humedad relativa %")
    viento: float = Field(..., ge=0, description="Velocidad del viento en km/h")
    visibilidad: float = Field(..., ge=0, description="Visibilidad en metros")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperatura": 15.5,
                "humedad": 85,
                "viento": 35,
                "visibilidad": 2500
            }
        }

class RiskResponse(BaseModel):
    riesgo: str = Field(..., description="Nivel de riesgo: Seguro, Precaución, Peligro")
    confianza: float = Field(..., description="Confianza de la predicción (0-1)")
    probabilidades: Dict[str, float] = Field(..., description="Probabilidades por clase")
    datos_clima: Dict = Field(..., description="Datos meteorológicos usados")
    
    class Config:
        json_schema_extra = {
            "example": {
                "riesgo": "Peligro",
                "confianza": 0.95,
                "probabilidades": {
                    "Seguro": 0.02,
                    "Precaución": 0.03,
                    "Peligro": 0.95
                },
                "datos_clima": {
                    "temperatura": 15.5,
                    "humedad": 85,
                    "viento": 35,
                    "visibilidad": 2500
                }
            }
        }
