"""
Motor de procesamiento batch para predicciones masivas.
"""
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session

from backend.services.ml_service_v2 import ml_service_v2
from backend.database import SessionLocal
from backend.batch.batch_config import BatchConfig, DEFAULT_CONFIG


logger = logging.getLogger(__name__)


class BatchPredictor:
    """Procesador de predicciones batch."""
    
    def __init__(self, config: Optional[BatchConfig] = None, ml_service=None):
        self.config = config or DEFAULT_CONFIG
        self.ml_service = ml_service  # Permitir inyectar servicio para tests
        self._setup_logging()
    
    def _setup_logging(self):
        """Configurar logging para batch."""
        log_file = self.config.log_dir / f"batch_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logging.basicConfig(
            level=logging.INFO if self.config.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def process_file(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        save_to_db: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Procesa un archivo individual con predicciones batch.
        
        Args:
            input_file: Ruta del archivo de entrada
            output_file: Ruta del archivo de salida (opcional)
            save_to_db: Si guardar en BD (usa config por defecto si None)
            
        Returns:
            DataFrame con predicciones
        """
        save_to_db = save_to_db if save_to_db is not None else self.config.save_to_db
        
        logger.info(f"📂 Procesando archivo: {input_file}")
        
        try:
            # Leer archivo de entrada
            df = self._read_input_file(input_file)
            logger.info(f"✅ Leídas {len(df)} filas")
            
            # Validar columnas requeridas
            self._validate_input(df)
            
            # Procesar en chunks si es necesario
            if len(df) > self.config.chunk_size:
                results = self._process_in_chunks(df, save_to_db)
            else:
                results = self._process_chunk(df, save_to_db)
            
            # Guardar resultados
            if output_file:
                self._save_results(results, output_file)
                logger.info(f"💾 Resultados guardados en: {output_file}")
            
            # Archivar archivo de entrada
            self._archive_file(input_file)
            
            logger.info(f"✅ Procesamiento completado: {len(results)} predicciones")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error procesando {input_file}: {str(e)}")
            self._move_to_error(input_file)
            raise
    
    def process_directory(self) -> List[pd.DataFrame]:
        """
        Procesa todos los archivos CSV en el directorio de entrada.
        
        Returns:
            Lista de DataFrames con resultados
        """
        input_files = list(self.config.input_dir.glob("*.csv"))
        
        if not input_files:
            logger.warning(f"⚠️  No se encontraron archivos en {self.config.input_dir}")
            return []
        
        logger.info(f"📁 Encontrados {len(input_files)} archivos para procesar")
        
        results = []
        for input_file in input_files:
            try:
                output_file = self.config.output_dir / f"predicted_{input_file.name}"
                result = self.process_file(input_file, output_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Error procesando {input_file}: {e}")
                continue
        
        logger.info(f"🎉 Procesamiento completado: {len(results)}/{len(input_files)} archivos")
        return results
    
    def _read_input_file(self, file_path: Path) -> pd.DataFrame:
        """Lee el archivo de entrada (soporta CSV, JSON, Excel)."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Formato no soportado: {suffix}")
    
    def _validate_input(self, df: pd.DataFrame):
        """Valida que el DataFrame tenga las columnas mínimas requeridas."""
        # Columnas que deben estar en español o inglés
        required_spanish = ["temperatura", "humedad", "presion", "viento", 
                          "rafaga", "visibilidad", "precipitacion", "nubes", "hielo"]
        required_english = ["temp", "humidity", "pressure", "wind_speed", 
                          "wind_gust", "visibility", "precipitation", "clouds", "ice_risk"]
        
        has_spanish = all(col in df.columns for col in required_spanish)
        has_english = all(col in df.columns for col in required_english)
        
        if not (has_spanish or has_english):
            raise ValueError(
                f"El archivo debe contener las columnas requeridas en español "
                f"{required_spanish} o inglés {required_english}"
            )
    
    def _process_in_chunks(self, df: pd.DataFrame, save_to_db: bool) -> pd.DataFrame:
        """Procesa el DataFrame en chunks."""
        logger.info(f"📊 Procesando en chunks de {self.config.chunk_size}")
        
        chunks = []
        total_chunks = (len(df) - 1) // self.config.chunk_size + 1
        
        for i in range(0, len(df), self.config.chunk_size):
            chunk_num = i // self.config.chunk_size + 1
            chunk = df.iloc[i:i + self.config.chunk_size]
            
            logger.info(f"⚙️  Procesando chunk {chunk_num}/{total_chunks}")
            result = self._process_chunk(chunk, save_to_db)
            chunks.append(result)
        
        return pd.concat(chunks, ignore_index=True)
    
    def _process_chunk(self, df: pd.DataFrame, save_to_db: bool) -> pd.DataFrame:
        """Procesa un chunk de datos."""
        db = None
        try:
            if save_to_db:
                db = SessionLocal()
            
            # Extraer ciudad e ICAO si existen
            ciudad = df['ciudad'].iloc[0] if 'ciudad' in df.columns else None
            icao = df['icao'].iloc[0] if 'icao' in df.columns else None
            
            # Usar el servicio inyectado o el global
            service = self.ml_service or ml_service_v2
            
            # Procesar predicciones
            results = service.predict_batch(
                df,
                ciudad=ciudad,
                icao=icao,
                db=db
            )
            
            return results
            
        finally:
            if db:
                db.close()
    
    def _save_results(self, df: pd.DataFrame, output_file: Path):
        """Guarda los resultados en el formato especificado."""
        output_format = self.config.output_format.lower()
        
        if output_format == 'csv':
            df.to_csv(output_file, index=False)
        elif output_format == 'json':
            df.to_json(output_file, orient='records', indent=2)
        elif output_format == 'parquet':
            df.to_parquet(output_file, index=False)
        else:
            raise ValueError(f"Formato no soportado: {output_format}")
    
    def _archive_file(self, file_path: Path):
        """Mueve el archivo procesado al directorio de archivo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        archive_path = self.config.archive_dir / archive_name
        
        file_path.rename(archive_path)
        logger.info(f"📦 Archivo archivado: {archive_path}")
    
    def _move_to_error(self, file_path: Path):
        """Mueve archivos con errores al directorio de errores."""
        if not file_path.exists():
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_name = f"{file_path.stem}_ERROR_{timestamp}{file_path.suffix}"
        error_path = self.config.error_dir / error_name
        
        file_path.rename(error_path)
        logger.error(f"⚠️  Archivo movido a errores: {error_path}")


# Instancia global
batch_predictor = BatchPredictor()