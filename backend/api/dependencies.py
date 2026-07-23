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
        X-API-Key: dev-key-123
    """
    if not settings.REQUIRE_API_KEY:
        return "public"
    
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key requerida. Incluir header: X-API-Key"
        )
    
    # 🔧 MEJORADO: Usa el método de settings
    valid_keys = settings.get_valid_api_keys()
    
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
    
    TODO: Implementar validación JWT real
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación requerido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # TODO: Implementar validación JWT
    # Por ahora retorna usuario dummy para desarrollo
    return {
        "user_id": "dummy_user",
        "email": "user@aerosafe.com",
        "role": "analyst"
    }


# ==================== VALIDACIONES ====================

async def validate_city_format(city: str) -> str:
    """
    Valida formato de ciudad (nombre,código_país)
    
    Formato esperado: "Ciudad,Código_País"
    Ejemplo válido: "Bogotá,CO"
    """
    if "," not in city:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato de ciudad inválido. Usar: 'Ciudad,Código_País' (ej: 'Bogotá,CO')"
        )
    
    city_name, country_code = city.split(",", 1)
    city_name = city_name.strip()
    country_code = country_code.strip().upper()
    
    if len(country_code) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código de país debe tener 2 caracteres (ej: CO, US, BR)"
        )
    
    if not country_code.isalpha():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código de país solo puede contener letras"
        )
    
    return f"{city_name},{country_code}"


async def validate_icao_code(icao: str) -> str:
    """
    Valida código ICAO de aeropuerto
    
    ICAO debe tener 4 caracteres alfabéticos
    Ejemplos válidos: SKBO (Bogotá), KJFK (New York JFK), EGLL (London Heathrow)
    """
    icao = icao.upper().strip()
    
    if len(icao) != 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código ICAO debe tener 4 caracteres (ej: SKBO, KJFK, EGLL)"
        )
    
    if not icao.isalpha():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código ICAO solo puede contener letras"
        )
    
    return icao


async def validate_latitude(lat: float) -> float:
    """Valida latitud (-90 a 90)"""
    if not -90 <= lat <= 90:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Latitud debe estar entre -90 y 90 grados"
        )
    return lat


async def validate_longitude(lon: float) -> float:
    """Valida longitud (-180 a 180)"""
    if not -180 <= lon <= 180:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Longitud debe estar entre -180 y 180 grados"
        )
    return lon


# ==================== RATE LIMITING ====================

class RateLimiter:
    """
    Rate limiter simple en memoria
    
    Nota: Para producción, usar Redis para compartir estado
    entre múltiples instancias de la API
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
        
        Args:
            key: Identificador único (ej: IP del cliente)
            max_requests: Máximo número de requests permitidos
            window_seconds: Ventana de tiempo en segundos
            
        Returns:
            True si está dentro del límite, False si lo excedió
        """
        import time
        
        current_time = time.time()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Limpiar requests fuera de la ventana de tiempo
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if current_time - req_time < window_seconds
        ]
        
        # Verificar si excedió el límite
        if len(self.requests[key]) >= max_requests:
            return False
        
        # Agregar request actual
        self.requests[key].append(current_time)
        return True
    
    def get_remaining_requests(self, key: str, max_requests: int = 100) -> int:
        """Retorna cuántos requests quedan disponibles"""
        if key not in self.requests:
            return max_requests
        return max(0, max_requests - len(self.requests[key]))


# Instancia global del rate limiter
rate_limiter = RateLimiter()


async def check_rate_limit(
    x_forwarded_for: Annotated[Optional[str], Header()] = None,
    x_real_ip: Annotated[Optional[str], Header()] = None
):
    """
    Middleware para verificar rate limit por IP
    
    Uso:
        @router.get("/endpoint", dependencies=[Depends(check_rate_limit)])
        def my_endpoint():
            return {"message": "ok"}
    """
    if not settings.RATE_LIMIT_ENABLED:
        return
    
    # Obtener IP del cliente (prioriza headers de proxy)
    client_ip = x_forwarded_for or x_real_ip or "unknown"
    
    if not rate_limiter.check_rate_limit(
        key=f"ip:{client_ip}",
        max_requests=settings.MAX_REQUESTS_PER_MINUTE,
        window_seconds=60
    ):
        remaining = rate_limiter.get_remaining_requests(
            f"ip:{client_ip}", 
            settings.MAX_REQUESTS_PER_MINUTE
        )
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Límite de requests excedido. Intenta de nuevo en 1 minuto. Restantes: {remaining}",
            headers={"Retry-After": "60"}
        )


# ==================== PAGINACIÓN ====================

class PaginationParams:
    """
    Parámetros de paginación reutilizables
    
    Attributes:
        page: Número de página (1-indexed)
        page_size: Cantidad de items por página
        skip: Offset para la query SQL
        limit: Límite de items a retornar
    """
    def __init__(self, page: int = 1, page_size: int = 20):
        self.page = max(1, page)
        self.page_size = min(100, max(1, page_size))  # Máximo 100 items
        self.skip = (self.page - 1) * self.page_size
        self.limit = self.page_size
    
    def get_pagination_info(self, total_items: int) -> dict:
        """
        Genera información de paginación para la respuesta
        
        Args:
            total_items: Total de items en la base de datos
            
        Returns:
            Dict con información de paginación
        """
        total_pages = (total_items + self.page_size - 1) // self.page_size
        
        return {
            "page": self.page,
            "page_size": self.page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": self.page < total_pages,
            "has_previous": self.page > 1
        }


async def get_pagination_params(
    page: int = 1,
    page_size: int = 20
) -> PaginationParams:
    """
    Dependencia para obtener parámetros de paginación
    
    Uso:
        @router.get("/items")
        def get_items(
            pagination: PaginationParams = Depends(get_pagination_params),
            db: Session = Depends(get_db)
        ):
            items = db.query(Item).offset(pagination.skip).limit(pagination.limit).all()
            total = db.query(Item).count()
            
            return {
                "items": items,
                "pagination": pagination.get_pagination_info(total)
            }
    """
    return PaginationParams(page=page, page_size=page_size)


# ==================== LOGGING ====================

async def log_request(
    request: str,
    x_forwarded_for: Annotated[Optional[str], Header()] = None,
    user_agent: Annotated[Optional[str], Header()] = None
):
    """
    Middleware para loguear información de las peticiones
    
    Uso como dependencia global en main.py:
        app = FastAPI(dependencies=[Depends(log_request)])
    """
    client_ip = x_forwarded_for or "unknown"
    logger.info(f"Request to {request} from IP: {client_ip}, User-Agent: {user_agent}")