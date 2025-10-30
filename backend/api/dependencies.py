from fastapi import Header, HTTPException, Depends, status
from typing import Optional, Annotated
from sqlalchemy.orm import Session
import logging

from core.config import settings
from database.connection import SessionLocal

logger = logging.getLogger(__name__)


# ==================== DATABASE ====================

def get_db():
    """
    Dependencia para obtener sesión de base de datos
    
    Uso:
        @router.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==================== AUTENTICACIÓN ====================

async def verify_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None
) -> str:
    """
    Verifica API key en headers (opcional para producción)
    
    Uso:
        @router.get("/protected")
        def protected_route(api_key: str = Depends(verify_api_key)):
            return {"message": "Authenticated"}
    
    Header requerido:
        X-API-Key: tu_api_key_aqui
    """
    if not settings.REQUIRE_API_KEY:
        return "public"
    
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key requerida. Incluir header: X-API-Key"
        )
    
    # Lista de API keys válidas (en producción usar BD o JWT)
    valid_keys = settings.VALID_API_KEYS.split(",")
    
    if x_api_key not in valid_keys:
        logger.warning(f"Intento de acceso con API key inválida: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    
    return x_api_key


async def get_current_user(
    authorization: Annotated[Optional[str], Header()] = None
) -> dict:
    """
    Obtiene usuario actual desde token JWT (para futuro)
    
    Uso:
        @router.get("/me")
        def get_me(user: dict = Depends(get_current_user)):
            return user
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación requerido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # TODO: Implementar validación JWT
    # Por ahora retorna usuario dummy
    return {
        "user_id": "dummy_user",
        "email": "user@example.com",
        "role": "user"
    }


# ==================== VALIDACIONES ====================

async def validate_city_format(city: str) -> str:
    """
    Valida formato de ciudad (nombre,código_país)
    
    Ejemplo válido: "Bogotá,CO"
    """
    if "," not in city:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato de ciudad inválido. Usar: 'Ciudad,Código_País' (ej: 'Bogotá,CO')"
        )
    
    city_name, country_code = city.split(",", 1)
    
    if len(country_code) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código de país debe tener 2 caracteres (ej: CO, US, BR)"
        )
    
    return city


async def validate_icao_code(icao: str) -> str:
    """
    Valida código ICAO de aeropuerto
    
    ICAO debe tener 4 caracteres (ej: SKBO)
    """
    icao = icao.upper().strip()
    
    if len(icao) != 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código ICAO debe tener 4 caracteres (ej: SKBO)"
        )
    
    if not icao.isalpha():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código ICAO solo puede contener letras"
        )
    
    return icao


# ==================== RATE LIMITING ====================

class RateLimiter:
    """
    Simple rate limiter en memoria (para producción usar Redis)
    """
    def __init__(self):
        self.requests = {}
    
    def check_rate_limit(
        self, 
        key: str, 
        max_requests: int = 100, 
        window_seconds: int = 60
    ) -> bool:
        """
        Verifica si se excedió el límite de requests
        
        Returns:
            True si está dentro del límite, False si lo excedió
        """
        import time
        
        current_time = time.time()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Limpiar requests antiguos
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if current_time - req_time < window_seconds
        ]
        
        # Verificar límite
        if len(self.requests[key]) >= max_requests:
            return False
        
        # Agregar request actual
        self.requests[key].append(current_time)
        return True


rate_limiter = RateLimiter()


async def check_rate_limit(
    x_forwarded_for: Annotated[Optional[str], Header()] = None
):
    """
    Verifica rate limit por IP
    
    Uso:
        @router.get("/endpoint", dependencies=[Depends(check_rate_limit)])
        def my_endpoint():
            return {"message": "ok"}
    """
    # Obtener IP del cliente
    client_ip = x_forwarded_for or "unknown"
    
    if not rate_limiter.check_rate_limit(
        key=f"ip:{client_ip}",
        max_requests=100,
        window_seconds=60
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demasiadas peticiones. Intenta de nuevo en 1 minuto."
        )


# ==================== PAGINACIÓN ====================

class PaginationParams:
    """
    Parámetros de paginación reutilizables
    """
    def __init__(
        self,
        page: int = 1,
        page_size: int = 20
    ):
        self.page = max(1, page)
        self.page_size = min(100, max(1, page_size))  # Máximo 100 items
        self.skip = (self.page - 1) * self.page_size
        self.limit = self.page_size


async def get_pagination_params(
    page: int = 1,
    page_size: int = 20
) -> PaginationParams:
    """
    Obtiene parámetros de paginación
    
    Uso:
        @router.get("/items")
        def get_items(pagination: PaginationParams = Depends(get_pagination_params)):
            return db.query(Item).offset(pagination.skip).limit(pagination.limit).all()
    """
    return PaginationParams(page=page, page_size=page_size)


# ==================== LOGGING ====================

async def log_request(
    x_forwarded_for: Annotated[Optional[str], Header()] = None,
    user_agent: Annotated[Optional[str], Header()] = None
):
    """
    Loguea información de la petición
    
    Uso como dependencia global en main.py
    """
    client_ip = x_forwarded_for or "unknown"
    logger.info(f"Request from IP: {client_ip}, User-Agent: {user_agent}")