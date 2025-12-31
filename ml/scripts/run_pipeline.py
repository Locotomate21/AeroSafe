import os
import sys
import subprocess
import time

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_script(script_path, description):
    """Ejecuta un script Python y maneja errores"""
    print_header(description)
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: No se encontró {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} completado exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR en {description}")
        print(f"Código de salida: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR inesperado: {e}")
        return False

def main():
    start_time = time.time()
    
    print_header("🚀 PIPELINE COMPLETO DE AEROSAFE")
    print("Este script ejecutará:")
    print("  1️⃣  Generación de dataset avanzado (30+ variables)")
    print("  2️⃣  Entrenamiento de modelos (LightGBM, RF, XGBoost, Ensemble)")
    print("  3️⃣  Logging en MLflow/DagsHub")
    print("  4️⃣  Visualizaciones y métricas\n")
    
    input("Presiona ENTER para comenzar...")
    
    # Verificar estructura de directorios
    print_header("📁 Verificando estructura de directorios")
    
    os.makedirs("data/dataset", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    
    print("✅ Directorios verificados")
    
    # Paso 1: Generar dataset
    success = run_script(
        "ml/scripts/generate_dataset_mejorado.py",
        "1️⃣  GENERANDO DATASET AVANZADO"
    )
    
    if not success:
        print("\n❌ Pipeline detenido: Error en generación de dataset")
        return 1
    
    time.sleep(2)
    
    # Paso 2: Entrenar modelos
    success = run_script(
        "ml/scripts/train_model_v2.py",
        "2️⃣  ENTRENANDO MODELOS AVANZADOS (v2)"
    )
    
    if not success:
        print("\n❌ Pipeline detenido: Error en entrenamiento")
        return 1
    
    # Resumen final
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print_header("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"⏱️  Tiempo total: {minutes}m {seconds}s")
    print(f"📂 Archivos generados:")
    print(f"  📊 Dataset: data/dataset/weather_risk_advanced.csv")
    print(f"  🤖 Modelos: data/models/*.pkl")
    print(f"  📈 Gráficas: data/models/*.png")
    print(f"  📝 Info: data/models/model_info_advanced.txt")
    
    print(f"\n🔗 Ver resultados en DagsHub:")
    print(f"  https://dagshub.com/Locotomate21/AeroSafe/experiments")
    
    print("\n" + "=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())