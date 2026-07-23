from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

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
    # Por defecto SQLite, pero puede sobrescribirse con .env para PostgreSQL
    DATABASE_URL: str = "sqlite:///./aerosafe.db"

    # ML Models
    # BASE_DIR es backend/, la raíz de la aplicación: todo (ml/, data/,
    # models/, logs/) vive bajo ese directorio, y desde ahí se ejecuta
    # tanto uvicorn como pytest.
    #   config.py -> core/ -> backend/
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: str = "models/production/model.pkl"
    SCALER_PATH: str = "models/production/scaler.pkl"
    ENCODER_PATH: str = "models/production/label_encoder.pkl"
    FEATURE_NAMES_PATH: str = "models/production/feature_names.txt"
    
    # Weather API
    WEATHER_API_KEY: Optional[str] = None
    # 🔒 SEGURIDAD: API key debe venir de .env, NO hardcodeada
    OPENWEATHER_API_KEY: Optional[str] = None  
    BASE_URL: str = "https://api.openweathermap.org/data/2.5/weather"
    
    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    LOG_MAX_BYTES: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Autenticación
    REQUIRE_API_KEY: bool = False
    VALID_API_KEYS: str = "dev-key-123,prod-key-456"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    MAX_REQUESTS_PER_MINUTE: int = 100
    
    # Batch Processing
    BATCH_INPUT_DIR: str = "data/batch/input"
    BATCH_OUTPUT_DIR: str = "data/batch/output"
    BATCH_ERROR_DIR: str = "data/batch/errors"
    BATCH_ARCHIVE_DIR: str = "data/batch/archive"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
    
    def get_origins_list(self) -> list[str]:
        """Convierte ALLOWED_ORIGINS string a lista"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    def get_model_path(self, relative_path: str) -> Path:
        """Obtiene ruta absoluta del modelo"""
        return self.BASE_DIR / relative_path
    
    def get_valid_api_keys(self) -> set[str]:
        """Obtiene conjunto de API keys válidas"""
        return {key.strip() for key in self.VALID_API_KEYS.split(",")}


settings = Settings()

# 🔧 Variables globales para compatibilidad con scripts antiguos
# IMPORTANTE: Estas pueden ser None si no están en .env
OPENWEATHER_API_KEY = settings.OPENWEATHER_API_KEY
BASE_URL = settings.BASE_URL