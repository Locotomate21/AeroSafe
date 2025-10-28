# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.risk_routes import router as risk_router

app = FastAPI(title="AeroSafe API", version="1.0")

# Configurar CORS (para permitir peticiones desde el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes cambiarlo a ["http://localhost:3000"] más adelante
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas
app.include_router(risk_router, prefix="/api/risk", tags=["Risk Evaluation"])

# Ruta base de prueba
@app.get("/")
def root():
    return {"message": "🚀 AeroSafe API funcionando correctamente"}

