import argparse
import sys
from pathlib import Path

# Agregar el directorio raíz al path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from batch.predict_batch import BatchPredictor
from batch.batch_config import BatchConfig


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar predicciones batch para AeroSafe"
    )
    
    # Modo de operación
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--file',
        type=str,
        help='Procesar un archivo específico'
    )
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='Procesar todos los archivos en el directorio de entrada'
    )
    
    # Opciones de configuración
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/batch/input',
        help='Directorio de entrada (default: data/batch/input)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/batch/output',
        help='Directorio de salida (default: data/batch/output)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Tamaño de chunks para procesar (default: 1000)'
    )
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='No guardar predicciones en la base de datos'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['csv', 'json', 'parquet'],
        default='csv',
        help='Formato de salida (default: csv)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso (menos logs)'
    )
    
    args = parser.parse_args()
    
    # Crear configuración personalizada
    config = BatchConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size,
        save_to_db=not args.no_db,
        output_format=args.output_format,
        verbose=not args.quiet
    )
    
    # Crear predictor
    predictor = BatchPredictor(config)
    
    try:
        if args.file:
            # Procesar archivo específico
            input_file = Path(args.file)
            if not input_file.exists():
                print(f"❌ Error: Archivo no encontrado: {input_file}")
                sys.exit(1)
            
            output_file = config.output_dir / f"predicted_{input_file.name}"
            predictor.process_file(input_file, output_file)
            
        else:
            # Procesar todos los archivos
            results = predictor.process_directory()
            print(f"\n✅ Procesamiento completado: {len(results)} archivos")
    
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()