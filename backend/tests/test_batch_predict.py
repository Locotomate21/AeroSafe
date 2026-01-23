import pytest
import pandas as pd
import tempfile
from pathlib import Path

from backend.batch.predict_batch import BatchPredictor
from backend.batch.batch_config import BatchConfig


@pytest.fixture
def temp_dirs():
    """Crea directorios temporales para tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        config = BatchConfig(
            input_dir=tmppath / "input",
            output_dir=tmppath / "output",
            archive_dir=tmppath / "archive",
            error_dir=tmppath / "errors",
            log_dir=tmppath / "logs",
            save_to_db=False  # No guardar en BD en tests
        )
        
        yield config


@pytest.fixture
def sample_batch_data():
    """Crea datos de ejemplo para batch - USAR FIXTURE ml_service para evitar conflictos."""
    return pd.DataFrame([
        {
            "ciudad": "Bogotá",
            "temperatura": 20.0,
            "humedad": 80.0,
            "presion": 1013.0,
            "viento": 7.0,
            "rafaga": 10.0,
            "visibilidad": 6000.0,
            "precipitacion": 0.0,
            "nubes": 50.0,
            "hielo": 0,
        },
    ])


def test_batch_predictor_initialization(temp_dirs):
    """Test que el predictor se inicializa correctamente."""
    predictor = BatchPredictor(temp_dirs)
    
    assert predictor.config == temp_dirs
    assert temp_dirs.input_dir.exists()
    assert temp_dirs.output_dir.exists()


def test_process_single_file(temp_dirs, sample_batch_data, ml_service):
    """Test procesar un archivo individual."""
    predictor = BatchPredictor(temp_dirs, ml_service=ml_service)
    
    # Crear archivo de entrada
    input_file = temp_dirs.input_dir / "test_input.csv"
    sample_batch_data.to_csv(input_file, index=False)
    
    # Procesar
    output_file = temp_dirs.output_dir / "test_output.csv"
    result = predictor.process_file(input_file, output_file)
    
    # Validar
    assert len(result) == 3
    assert "riesgo" in result.columns
    assert "confianza" in result.columns
    assert output_file.exists()
    
    # Validar que el archivo se archivó
    archived_files = list(temp_dirs.archive_dir.glob("test_input_*.csv"))
    assert len(archived_files) == 1


def test_process_in_chunks(temp_dirs, ml_service):
    """Test procesar datos en chunks."""
    # Crear predictor con chunk_size pequeño
    temp_dirs.chunk_size = 2
    predictor = BatchPredictor(temp_dirs, ml_service=ml_service)
    
    # Crear datos grandes
    large_data = pd.DataFrame([
        {
            "ciudad": f"Ciudad{i}",
            "temperatura": 20.0 + i,
            "humedad": 80.0,
            "presion": 1013.0,
            "viento": 7.0,
            "rafaga": 10.0,
            "visibilidad": 6000.0,
            "precipitacion": 0.0,
            "nubes": 50.0,
            "hielo": 0,
        }
        for i in range(5)
    ])
    
    # Guardar y procesar
    input_file = temp_dirs.input_dir / "large_input.csv"
    large_data.to_csv(input_file, index=False)
    
    result = predictor.process_file(input_file)
    
    # Validar
    assert len(result) == 5
    assert "riesgo" in result.columns


def test_process_directory(temp_dirs, sample_batch_data, ml_service):
    """Test procesar múltiples archivos en un directorio."""
    predictor = BatchPredictor(temp_dirs, ml_service=ml_service)
    
    # Crear múltiples archivos
    for i in range(3):
        input_file = temp_dirs.input_dir / f"test_input_{i}.csv"
        sample_batch_data.to_csv(input_file, index=False)
    
    # Procesar directorio
    results = predictor.process_directory()
    
    # Validar
    assert len(results) == 3
    assert all(len(r) == 3 for r in results)
    
    # Validar archivos de salida
    output_files = list(temp_dirs.output_dir.glob("predicted_*.csv"))
    assert len(output_files) == 3


def test_invalid_input_format(temp_dirs):
    """Test manejo de formato inválido."""
    predictor = BatchPredictor(temp_dirs)  # No necesita ml_service para este test
    
    # Crear archivo con formato no soportado
    invalid_file = temp_dirs.input_dir / "test.txt"
    invalid_file.write_text("invalid data")
    
    # Debe fallar
    with pytest.raises(ValueError, match="Formato no soportado"):
        predictor.process_file(invalid_file)
    
    # El archivo debe moverse a errores
    error_files = list(temp_dirs.error_dir.glob("test_ERROR_*.txt"))
    assert len(error_files) == 1


def test_missing_columns(temp_dirs, ml_service):
    """Test validación de columnas faltantes."""
    predictor = BatchPredictor(temp_dirs, ml_service=ml_service)
    
    # Datos sin columnas requeridas
    invalid_data = pd.DataFrame([
        {"ciudad": "Bogotá", "temperatura": 20.0}
    ])
    
    input_file = temp_dirs.input_dir / "invalid.csv"
    invalid_data.to_csv(input_file, index=False)
    
    # Debe fallar
    with pytest.raises(ValueError, match="debe contener las columnas requeridas"):
        predictor.process_file(input_file)


def test_output_formats(temp_dirs, sample_batch_data, ml_service):
    """Test diferentes formatos de salida."""
    for output_format in ['csv', 'json']:
        temp_dirs.output_format = output_format
        predictor = BatchPredictor(temp_dirs, ml_service=ml_service)
        
        input_file = temp_dirs.input_dir / f"test_{output_format}.csv"
        sample_batch_data.to_csv(input_file, index=False)
        
        output_file = temp_dirs.output_dir / f"output.{output_format}"
        result = predictor.process_file(input_file, output_file)
        
        assert output_file.exists()
        assert len(result) == 3