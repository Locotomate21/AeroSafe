import os
import json
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join("data", "weather_dataset.csv")

def consolidate_weather_data():
    records = []
    for file in os.listdir(RAW_DIR):
        if file.endswith(".json"):
            with open(os.path.join(RAW_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append(data)

    if records:
        df = pd.DataFrame(records)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Dataset consolidado: {OUTPUT_FILE}")
    else:
        print("⚠️ No se encontraron archivos JSON en data/raw")

if __name__ == "__main__":
    consolidate_weather_data()
