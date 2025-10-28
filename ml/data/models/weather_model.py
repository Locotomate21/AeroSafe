from pydantic import BaseModel

class WeatherData(BaseModel):
    ciudad: str
    pais: str
    fecha: str
    temperatura: float
    humedad: int
    presion: int
    viento: float
    visibilidad: int
    descripcion: str
