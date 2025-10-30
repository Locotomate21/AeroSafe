from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes.risk_routes import router as risk_router
from api.routes.weather_routes import router as weather_router
from core.config import settings
from core.logging import setup_logging
from database.connection import init_db

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API de predicción de riesgo meteorológico para aviación"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_origins_list(),  # Usar método para obtener lista
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(risk_router, prefix="/api/v1/risk", tags=["risk"])
app.include_router(weather_router, prefix="/api/v1/weather", tags=["weather"])

@app.get("/")
async def root():
    return {
        "message": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AeroSafe API"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )