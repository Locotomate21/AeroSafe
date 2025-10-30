from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API
    PROJECT_NAME: str = "AeroSafe"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./weather.db"
    
    # ML Models
    MODEL_PATH: str = "../ml/data/models/risk_model.pkl"
    SCALER_PATH: str = "../ml/data/models/scaler.pkl"
    ENCODER_PATH: str = "../ml/data/models/label_encoder.pkl"
    
    # Weather API
    WEATHER_API_KEY: Optional[str] = None
    OPENWEATHER_API_KEY: str = "a45d492668dceb132d0d67106b718810"
    BASE_URL: str = "https://api.openweathermap.org/data/2.5/weather"
    
    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
    
    def get_origins_list(self):
        """Convierte ALLOWED_ORIGINS string a lista"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    # Autenticación
    REQUIRE_API_KEY: bool = False
    VALID_API_KEYS: str = "dev-key-123,prod-key-456"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    MAX_REQUESTS_PER_MINUTE: int = 100


settings = Settings()

# 🔧 Agregamos variables globales para compatibilidad con scripts antiguos
OPENWEATHER_API_KEY = settings.OPENWEATHER_API_KEY
BASE_URL = settings.BASE_URL
