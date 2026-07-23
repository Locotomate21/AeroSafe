import requests
from backend.core.config import settings

url = f"{settings.BASE_URL}?q=Bogotá&appid={settings.OPENWEATHER_API_KEY}&units=metric"

try:
    response = requests.get(url)
    response.raise_for_status()
    print("✅ Conexión a OpenWeather OK")
    print(response.json())
except requests.HTTPError as e:
    print(f"❌ Error HTTP: {e}")
