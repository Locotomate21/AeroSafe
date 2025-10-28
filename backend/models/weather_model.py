# backend/models/weather_model.py
from pydantic import BaseModel
from datetime import datetime

class WeatherData(BaseModel):
    city: str
    temperature: float
    humidity: int
    description: str
    timestamp: datetime
