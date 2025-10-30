from datetime import datetime
from backend.models.weather_model import WeatherData
from backend.services.risk_predictor import evaluate_risk
from backend.models.database import SessionLocal, Weather

# --- Adaptador: convierte WeatherData a formato de evaluate_risk ---
def weatherdata_to_risk_input(weather: WeatherData, viento=0, visibilidad=10000):
    return {
        "temperatura": weather.temperature,
        "viento": viento,
        "visibilidad": visibilidad,
        "humedad": weather.humidity,
        "descripcion": weather.description
    }

# --- Datos de ejemplo ---
weather_example = WeatherData(
    city="Bogotá",
    temperature=20.5,
    humidity=60,
    description="nublado",
    timestamp=datetime.now()
)

# --- Transformar a formato de riesgo ---
risk_input = weatherdata_to_risk_input(weather_example, viento=12, visibilidad=5000)

# --- Evaluar riesgo ---
resultado = evaluate_risk(risk_input)

# --- Guardar en la base de datos ---
db = SessionLocal()
weather_record = Weather(
    city=weather_example.city,
    temperature=weather_example.temperature,
    humidity=weather_example.humidity,
    description=weather_example.description,
    timestamp=weather_example.timestamp
)
db.add(weather_record)
db.commit()
db.close()

# --- Mostrar resultado ---
print("=== Datos meteorológicos ===")
print(risk_input)
print("\n=== Evaluación de riesgo ===")
print(resultado)