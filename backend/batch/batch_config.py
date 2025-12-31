"""
Configuración para procesamiento batch de predicciones.
"""
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class BatchConfig:
    """Configuración para procesamiento batch."""
    
    # Rutas
    input_dir: Path = Path("data/batch/input")
    output_dir: Path = Path("data/batch/output")
    archive_dir: Path = Path("data/batch/archive")
    error_dir: Path = Path("data/batch/errors")
    
    # Tamaño de chunks para procesar
    chunk_size: int = 1000
    
    # Guardar en base de datos
    save_to_db: bool = True
    
    # Formato de salida
    output_format: str = "csv"  # csv, json, parquet
    
    # Logging
    log_dir: Path = Path("logs/batch")
    verbose: bool = True
    
    def __post_init__(self):
        """Crear directorios si no existen."""
        for directory in [
            self.input_dir,
            self.output_dir,
            self.archive_dir,
            self.error_dir,
            self.log_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Configuración por defecto
DEFAULT_CONFIG = BatchConfig()