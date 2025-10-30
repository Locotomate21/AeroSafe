from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Definir Base UNA SOLA VEZ aquí
Base = declarative_base()


class WeatherRecord(Base):
    """Registro de datos meteorológicos"""
    __tablename__ = "weather_records"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Identificación
    icao = Column(String(4), index=True, nullable=True)
    ciudad = Column(String(100), index=True)
    pais = Column(String(50))
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Variables principales
    temperatura = Column(Float)
    humedad = Column(Float)
    viento = Column(Float)
    visibilidad = Column(Float)
    
    # Variables adicionales
    presion = Column(Float, nullable=True)
    punto_rocio = Column(Float, nullable=True)
    nubes = Column(Integer, nullable=True)
    viento_direccion = Column(Integer, nullable=True)
    viento_rafagas = Column(Float, nullable=True)
    
    # Condiciones
    descripcion = Column(String(200), nullable=True)
    condicion = Column(String(50), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class RiskPrediction(Base):
    """Predicción de riesgo"""
    __tablename__ = "risk_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Identificación
    icao = Column(String(4), index=True, nullable=True)
    ciudad = Column(String(100))
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Predicción
    riesgo = Column(String(20), index=True)  # Seguro, Precaución, Peligro
    confianza = Column(Float)
    probabilidades = Column(JSON)  # {"Seguro": 0.1, "Precaución": 0.2, "Peligro": 0.7}
    
    # Datos usados
    temperatura = Column(Float)
    humedad = Column(Float)
    viento = Column(Float)
    visibilidad = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class Airport(Base):
    """Información de aeropuertos"""
    __tablename__ = "airports"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Identificación
    icao = Column(String(4), unique=True, index=True)
    iata = Column(String(3), nullable=True)
    nombre = Column(String(200))
    ciudad = Column(String(100))
    pais = Column(String(50))
    
    # Ubicación
    latitud = Column(Float)
    longitud = Column(Float)
    elevacion = Column(Float, nullable=True)
    
    # Metadata
    activo = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Weather(Base):
    """Modelo legacy (mantener por compatibilidad si se usa en otro lado)"""
    __tablename__ = "weather"

    id = Column(Integer, primary_key=True, index=True)
    city = Column(String)
    temperature = Column(Float)
    humidity = Column(Integer)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)