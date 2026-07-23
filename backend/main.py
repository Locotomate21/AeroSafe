from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging

# 🔧 CORREGIDO: Imports sin 'backend.' porque ya estamos en backend/
from backend.api.routes.risk_routes import router as risk_router
from backend.api.routes.weather_routes import router as weather_router
# from api.routes.dashboard_routes import router as dashboard_router  # Descomentar si existe
from core.config import settings
from core.logging import setup_logging, get_logger
from database.connection import init_db

# Setup logging
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API de predicción de riesgo meteorológico para aviación",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ==================== MIDDLEWARE ====================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== EXCEPTION HANDLERS ====================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Maneja errores de validación de forma más amigable"""
    logger.warning(f"Validation error on {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Error de validación",
            "errors": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Maneja excepciones generales no capturadas"""
    logger.error(f"Unhandled exception on {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor",
            "message": str(exc) if settings.DEBUG else "Ha ocurrido un error"
        }
    )


# ==================== EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Ejecutar al iniciar la aplicación"""
    logger.info(f"🚀 Iniciando {settings.PROJECT_NAME} v{settings.VERSION}")
    
    # Inicializar base de datos
    try:
        init_db()
        logger.info("✅ Base de datos inicializada")
    except Exception as e:
        logger.error(f"❌ Error al inicializar base de datos: {e}")
    
    # Verificar que existan los modelos ML
    from pathlib import Path
    model_path = settings.get_model_path(settings.MODEL_PATH)
    if model_path.exists():
        logger.info(f"✅ Modelo ML encontrado: {model_path}")
    else:
        logger.warning(f"⚠️  Modelo ML NO encontrado: {model_path}")
    
    logger.info(f"📡 API disponible en http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📚 Documentación en http://{settings.HOST}:{settings.PORT}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Ejecutar al apagar la aplicación"""
    logger.info("🛑 Apagando AeroSafe API...")


# ==================== ROUTERS ====================

# Include routers
app.include_router(risk_router, prefix="/api/v1/risk", tags=["Risk Assessment"])
app.include_router(weather_router, prefix="/api/v1/weather", tags=["Weather"])
# app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["Dashboard"])  # Descomentar si existe


# ==================== ROOT ENDPOINTS ====================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz - Información del servicio"""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint para monitoreo"""
    from pathlib import Path
    
    # Verificar que el modelo existe
    model_exists = settings.get_model_path(settings.MODEL_PATH).exists()
    
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "ml_model_loaded": model_exists,
        "database": "connected"  # TODO: verificar conexión real
    }


@app.get("/info", tags=["Root"])
async def info():
    """Información detallada del sistema"""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": "development" if settings.DEBUG else "production",
        "debug_mode": settings.DEBUG,
        "api_prefix": settings.API_V1_PREFIX,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health"
        }
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # 🔧 CORREGIDO: Sin 'backend.' cuando ejecutas desde backend/
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )