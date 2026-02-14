"""
Utilidades para conexión y operaciones con la base de datos usando SQLAlchemy.
"""
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from .logger import get_logger
from .env import env
from .models import Base, Person, Report

logger = get_logger(__name__)


class SupabaseDatabase:
    """
    Clase para gestionar la conexión y operaciones con la base de datos PostgreSQL (Supabase).
    Implementa el patrón Singleton para garantizar una única instancia de conexión.
    """

    _instance: Optional['SupabaseDatabase'] = None

    def __new__(cls):
        """Implementación del patrón Singleton"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Inicializa la conexión a la base de datos"""
        if not hasattr(self, '_initialized'):
            self._db_user = env['DB_USER']
            self._db_password = env['DB_PASSWORD']
            self._db_host = env['DB_HOST']
            self._db_port = env['DB_PORT']
            self._db_name = env['DB_NAME']

            self._db_url = self._build_connection_url()
            self._engine = self._create_engine()
            self._session_local = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
            self._initialized = True

            logger.info(f"Database connection initialized: {self._db_host}:{self._db_port}/{self._db_name}")

    def _build_connection_url(self) -> str:
        """
        Construye la URL de conexión a la base de datos.

        Returns:
            URL de conexión en formato PostgreSQL
        """
        return (
            f"postgresql+psycopg2://{self._db_user}:{self._db_password}"
            f"@{self._db_host}:{self._db_port}/{self._db_name}?sslmode=require"
        )

    def _create_engine(self) -> Engine:
        """
        Crea el engine de SQLAlchemy con configuración optimizada.

        Returns:
            Engine de SQLAlchemy configurado
        """
        return create_engine(
            self._db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            connect_args={
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
        )

    @property
    def engine(self) -> Engine:
        """Retorna el engine de SQLAlchemy"""
        return self._engine

    @property
    def session_local(self) -> sessionmaker:
        """Retorna el sessionmaker configurado"""
        return self._session_local

    @property
    def db_url(self) -> str:
        """Retorna la URL de conexión a la base de datos"""
        return self._db_url

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager para obtener una sesión de base de datos.
        La sesión se commitea automáticamente si no hay errores,
        o se hace rollback en caso de excepción.

        Yields:
            Sesión de SQLAlchemy

        Example:
            with db.get_session() as session:
                person = session.query(Person).filter_by(id=1).first()
        """
        session = self._session_local()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def get_db_dependency(self) -> Generator[Session, None, None]:
        """
        Dependency para FastAPI que proporciona una sesión de base de datos.
        No hace commit automático, FastAPI maneja el ciclo de vida.

        Yields:
            Sesión de SQLAlchemy

        Example:
            @app.get("/items")
            def read_items(db: Session = Depends(database.get_db_dependency)):
                return db.query(Item).all()
        """
        session = self._session_local()
        try:
            yield session
        finally:
            session.close()

    def init_db(self):
        """
        Inicializa la base de datos creando todas las tablas.
        Solo crea las tablas que no existen.
        """
        try:
            Base.metadata.create_all(bind=self._engine)
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def close(self):
        """
        Cierra todas las conexiones del pool de la base de datos.
        Útil para cleanup en tests o shutdown de la aplicación.
        """
        if hasattr(self, '_engine'):
            self._engine.dispose()
            logger.info("Database connections closed")


# Instancia global de la base de datos (Singleton)
database = SupabaseDatabase()

# Variables y funciones de compatibilidad con código legacy
engine = database.engine
SessionLocal = database.session_local


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager para obtener una sesión de base de datos (compatibilidad legacy).

    Yields:
        Sesión de SQLAlchemy
    """
    with database.get_session() as session:
        yield session


def get_db() -> Generator[Session, None, None]:
    """
    Dependency para FastAPI (compatibilidad legacy).

    Yields:
        Sesión de SQLAlchemy
    """
    yield from database.get_db_dependency()


def init_db():
    """
    Inicializa la base de datos (compatibilidad legacy).
    """
    database.init_db()
