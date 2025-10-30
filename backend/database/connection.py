from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import settings

# Importar Base desde models/database.py (NO redefinirla aquí)
from models.database import Base

# Crear engine usando configuración
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Inicializa la base de datos (crea todas las tablas)"""
    Base.metadata.create_all(bind=engine)
    print("✅ Base de datos inicializada")

def get_db():
    """
    Generador de sesiones de BD
    Uso en FastAPI: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()