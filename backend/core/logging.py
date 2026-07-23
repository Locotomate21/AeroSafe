import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from core.config import settings

def setup_logging(name: str = __name__) -> logging.Logger:
    """
    Configura el sistema de logging con rotación de archivos
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    
    # Resolver LOG_FILE contra BASE_DIR: si no, la ruta relativa del .env
    # depende del cwd y los logs terminan en sitios distintos según se
    # arranque uvicorn, pytest o un script de ml/.
    log_file = Path(settings.LOG_FILE)
    if not log_file.is_absolute():
        log_file = settings.BASE_DIR / log_file

    # Crear carpeta logs si no existe
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Obtener logger
    logger = logging.getLogger(name)
    
    # Evitar duplicar handlers si ya está configurado
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Formato de logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para archivo con rotación
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    file_handler.setFormatter(formatter)
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Agregar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging configurado correctamente - Nivel: {settings.LOG_LEVEL}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger ya configurado o crea uno nuevo
    
    Args:
        name: Nombre del módulo (usar __name__)
        
    Returns:
        Logger configurado
    """
    return setup_logging(name)