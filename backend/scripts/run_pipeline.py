# backend/scripts/run_pipeline.py
import sys, os
import subprocess
from datetime import datetime

# === Configuración de rutas ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "data", "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
LOG_FILE = os.path.join(BASE_DIR, "backend", "log.txt")

# Asegurar que el backend esté en el path
sys.path.append(BASE_DIR)


def log(msg):
    """Registra mensajes tanto en consola como en archivo de log."""
    safe_msg = msg.encode("ascii", errors="ignore").decode()  # Evita errores de codificación (emoji)
    print(safe_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {safe_msg}\n")


def latest_file_time(path):
    """Devuelve la fecha del archivo más reciente en el directorio dado."""
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    if not files:
        return None
    return max(os.path.getmtime(f) for f in files)


def latest_model_time():
    """Devuelve la fecha del modelo más reciente."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        return 0
    models = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not models:
        return 0
    return max(os.path.getmtime(m) for m in models)


def run(script):
    """Ejecuta un script del módulo backend.scripts y captura su salida."""
    log(f"== Ejecutando {script} ==")
    try:
        result = subprocess.run(
            ["python", "-m", f"backend.scripts.{script}"],
            capture_output=True,
            text=True,
            cwd=BASE_DIR  # Asegura que se ejecute desde la raíz del proyecto
        )
        if result.stdout:
            log(result.stdout.strip())
        if result.stderr:
            log(f"⚠️ Error en {script}: {result.stderr.strip()}")
    except Exception as e:
        log(f"❌ No se pudo ejecutar {script}: {e}")


if __name__ == "__main__":
    log("\n=== 🚀 Iniciando pipeline AeroSafe ===")

    # Paso 1: Recolectar datos
    run("collect_weather_data")

    # Paso 2: Revisar si hay nuevos datos
    last_data = latest_file_time(DATA_DIR)
    last_model = latest_model_time()

    if not last_data:
        log("❌ No se encontraron archivos de dataset.")
    elif last_data > last_model:
        log("📈 Nuevos datos detectados, reentrenando modelo...")
        run("train_model_v2")
    else:
        log("⏩ Modelo ya está actualizado. No se requiere reentrenamiento.")

    # Paso 3: Ejecutar evaluación de riesgo
    run("test_risk_pipeline")

    log("✅ Pipeline completado con éxito.")
