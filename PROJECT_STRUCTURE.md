AeroSafe/
├── Dockerfile
├── PROJECT_STRUCTURE.md
├── run_aerosafe.bat
├── run_pipeline.py
├── run_training.bat
├── show_structure.py
├── weather.db
├── .dvc/ # Configuración de DVC para control de datos y modelos
│ ├── config
│ ├── cache/
│ └── tmp/
├── backend/ # API principal (FastAPI)
│ ├── main.py
│ ├── docker-compose.yml
│ ├── core/
│ │ ├── config.py
│ │ └── logging.py
│ ├── api/
│ │ └── routes/
│ │ ├── weather_routes.py
│ │ └── risk_routes.py
│ ├── database/
│ │ ├── connection.py
│ │ └── repositories/
│ │ └── weather_repository.py
│ ├── models/
│ │ ├── schemas.py
│ │ ├── weather_model.py
│ │ └── risk_model_loader.py
│ ├── services/
│ │ ├── weather_service.py
│ │ ├── weather_api_service.py
│ │ ├── risk_predictor.py
│ │ └── ml_service.py
│ ├── scripts/
│ │ ├── collect_weather_data.py
│ │ ├── check_db.py
│ │ └── test_risk_pipeline.py
│ ├── utils/
│ │ ├── helpers.py
│ │ └── weather_data.py
│ └── test/
│ ├── conftest.py
│ ├── test_api/
│ │ ├── test_health.py
│ │ ├── test_weather_routes.py
│ │ └── test_risk_routes.py
│ └── test_services/
│ └── test_ml_service.py
│
├── ml/ # Modelado y entrenamiento
│ ├── scripts/
│ │ ├── generate_dataset_mejorado.py
│ │ ├── train_model_v2.py
│ │ ├── train_model_v3.py
│ │ ├── fetch_real_weather.py
│ │ ├── preprocess_weather_data.py
│ │ └── collect_airport_history.py
│ ├── data/
│ │ ├── dataset/
│ │ │ └── weather_risk_advanced.csv
│ │ ├── models/
│ │ │ ├── risk_model.pkl
│ │ │ ├── risk_model_rf.pkl
│ │ │ ├── risk_model_xgb.pkl
│ │ │ ├── risk_model_ensemble.pkl
│ │ │ ├── scaler.pkl
│ │ │ ├── label_encoder.pkl
│ │ │ └── model_info.txt
│ │ ├── raw/ # Datos meteorológicos descargados por ciudad
│ │ └── processed/ # Archivos limpios y listos para entrenamiento
│ ├── notebooks/
│ │ ├── data_exploration.ipynb
│ │ └── model_training.ipynb
│ └── project_structure.txt
│
├── data/
│ ├── dataSet/
│ │ └── weather_risk_advanced.csv
│ └── models/
│ ├── risk_model.pkl
│ ├── scaler.pkl
│ ├── label_encoder.pkl
│ └── model_comparison.png
└── test_mlflow.py # Pruebas con el tracking de MLflow