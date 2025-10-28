@echo off
REM === Script para ejecutar el pipeline de AeroSafe ===
cd /d "C:\Users\USER\Desktop\aerosafe"

REM Activar entorno virtual
call venv\Scripts\activate

REM Ejecutar el pipeline completo
python -m backend.scripts.run_pipeline

REM Desactivar entorno virtual
deactivate

echo.
echo === Pipeline completado ===
pause

