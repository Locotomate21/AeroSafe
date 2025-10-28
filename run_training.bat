@echo off
echo ============================= >> log_train.txt
echo Entrenando modelo AEROSAFE - %date% %time% >> log_train.txt

cd /d C:\Users\USER\Desktop\aerosafe
C:\Users\USER\Desktop\aerosafe\venv\Scripts\python.exe -m backend.scripts.train_model_v2 >> log_train.txt 2>&1

echo Entrenamiento completado a las %time% >> log_train.txt
