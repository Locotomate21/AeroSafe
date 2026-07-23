from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from api.dependencies import rate_limiter, verify_api_key
from api.routes.risk_routes import router as risk_router
from api.routes.weather_routes import router as weather_router
from api.routes.dashboard_routes import router as dashboard_router
from core.config import settings
from core.logging import get_logger
from database.connection import init_db

# Setup logging
logger = get_logger(__name__)

# Rutas exentas de rate limiting: el health check lo consulta el
# orquestador cada pocos segundos y no debe consumir la cuota del cliente.
RUTAS_SIN_LIMITE = {"/health", "/docs", "/redoc", "/openapi.json"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación (reemplaza a @app.on_event)."""
    # --- Arranque ---
    logger.info("Iniciando %s v%s", settings.PROJECT_NAME, settings.VERSION)

    # Auditoría de seguridad: no aborta el arranque (rompería el desarrollo
    # local) pero deja cada punto débil en el log, para que un despliegue
    # inseguro no pase inadvertido.
    for problema in settings.validate_security():
        logger.warning("SEGURIDAD: %s", problema)

    logger.info(
        "Autenticación por API key: %s | Rate limit: %s",
        "activa" if settings.REQUIRE_API_KEY else "DESACTIVADA",
        f"{settings.MAX_REQUESTS_PER_MINUTE}/min" if settings.RATE_LIMIT_ENABLED else "DESACTIVADO",
    )

    try:
        init_db()
        logger.info("Base de datos inicializada")
    except Exception as e:
        logger.error("Error al inicializar base de datos: %s", e)

    # El estado del modelo se registra al arrancar, no cuando llega la
    # primera petición: si arranca en modo mock hay que enterarse ya.
    from services.ml_service_v2 import ml_service_v2

    if ml_service_v2 is not None and ml_service_v2.can_infer():
        logger.info("Modelo ML operativo")
    else:
        logger.error(
            "Modelo ML NO disponible: la API responderá con predicciones "
            "heurísticas marcadas como model_status='mock'"
        )

    logger.info("API disponible en http://%s:%s", settings.HOST, settings.PORT)
    logger.info("Documentación en http://%s:%s/docs", settings.HOST, settings.PORT)

    yield

    # --- Apagado ---
    logger.info("Apagando AeroSafe API")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API de predicción de riesgo meteorológico para aviación",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ==================== MIDDLEWARE ====================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_origins_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Limita las peticiones por IP.

    RateLimiter llevaba implementado en api/dependencies.py desde el
    principio, junto con RATE_LIMIT_ENABLED y MAX_REQUESTS_PER_MINUTE en
    la configuración, pero nunca se conectó a nada: la API aceptaba
    peticiones sin límite mientras aparentaba tener control de caudal.

    Nota: el contador vive en memoria del proceso. Con varias réplicas
    cada una lleva su propia cuenta; para eso hace falta Redis.
    """
    if not settings.RATE_LIMIT_ENABLED or request.url.path in RUTAS_SIN_LIMITE:
        return await call_next(request)

    cliente = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or request.headers.get("x-real-ip")
        or (request.client.host if request.client else "unknown")
    )

    if not rate_limiter.check_rate_limit(
        key=f"ip:{cliente}",
        max_requests=settings.MAX_REQUESTS_PER_MINUTE,
        window_seconds=60,
    ):
        logger.warning("Rate limit excedido para %s en %s", cliente, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": (
                    f"Límite de {settings.MAX_REQUESTS_PER_MINUTE} peticiones "
                    f"por minuto excedido."
                )
            },
            headers={"Retry-After": "60"},
        )

    return await call_next(request)


# ==================== EXCEPTION HANDLERS ====================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Maneja errores de validación de forma más amigable"""
    logger.warning(f"Validation error on {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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


# ==================== ROUTERS ====================

# Los routers de negocio exigen API key cuando REQUIRE_API_KEY=true.
# verify_api_key deja pasar todo si está desactivada, así el desarrollo
# local no necesita cabeceras.
protegido = [Depends(verify_api_key)]

app.include_router(
    risk_router, prefix="/api/v1/risk", tags=["Risk Assessment"],
    dependencies=protegido,
)
app.include_router(
    weather_router, prefix="/api/v1/weather", tags=["Weather"],
    dependencies=protegido,
)
app.include_router(
    dashboard_router, prefix="/api/v1/dashboard", tags=["Dashboard"],
    dependencies=protegido,
)


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
    """
    Health check para monitoreo.

    Comprueba el estado real de cada componente. Antes reportaba
    ml_model_loaded=true con solo verificar que el fichero .pkl existiera,
    y database='connected' sin abrir una conexión: podía devolver
    'healthy' mientras la API respondía con predicciones heurísticas.
    """
    from sqlalchemy import text

    from database.connection import engine
    from services.ml_service_v2 import ml_service_v2

    # El modelo está "cargado" solo si además se puede inferir con él
    # (modelo + scaler + encoders).
    modelo_ok = ml_service_v2 is not None and ml_service_v2.can_infer()

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_estado = "connected"
    except Exception as e:
        logger.error("Health check: fallo de base de datos: %s", e)
        db_estado = "error"

    saludable = modelo_ok and db_estado == "connected"

    return JSONResponse(
        status_code=status.HTTP_200_OK if saludable else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "healthy" if saludable else "degraded",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "ml_model_loaded": modelo_ok,
            "database": db_estado,
        },
    )


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
    # Ejecutar desde backend/:  python main.py
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )