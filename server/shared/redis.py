"""
Adapter para Redis que desacopla la librería redis del resto del proyecto.
Implementa el patrón Adapter para abstraer las operaciones de Redis.
"""
import json
from typing import Optional, Any
import redis

from .logger import get_logger
from .env import env

logger = get_logger(__name__)


class RedisAdapter:
    """
    Adapter para operaciones básicas de Redis.
    Abstrae la librería redis-py del resto del proyecto.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """
        Inicializa el adaptador de Redis.

        Args:
            host: Host del servidor Redis
            port: Puerto del servidor Redis
            db: Base de datos de Redis a usar
        """
        self.host = host
        self.port = port
        self.db = db
        self._client: Optional[redis.Redis] = None

    def connect(self) -> None:
        """
        Establece la conexión con Redis.

        Raises:
            ConnectionError: Si no se puede conectar a Redis
        """
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False
            )
            self._client.ping()
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Could not connect to Redis: {e}")

    def disconnect(self) -> None:
        """Cierra la conexión con Redis."""
        if self._client:
            self._client.close()

    def get(self, key: str) -> Optional[bytes]:
        """
        Obtiene un valor de Redis.

        Args:
            key: Clave a buscar

        Returns:
            Valor en bytes o None si no existe
        """
        if not self._client:
            logger.error("Redis client not connected when trying to get key")
            raise RuntimeError("Redis client not connected")
        return self._client.get(key)

    def set(self, key: str, value: bytes, expire_seconds: Optional[int] = None) -> bool:
        """
        Establece un valor en Redis.

        Args:
            key: Clave a establecer
            value: Valor en bytes a guardar
            expire_seconds: Segundos hasta que expire la clave (opcional)

        Returns:
            True si fue exitoso
        """
        if not self._client:
            logger.error("Redis client not connected when trying to set key")
            raise RuntimeError("Redis client not connected")

        result = self._client.set(key, value)

        if expire_seconds:
            self._client.expire(key, expire_seconds)

        return bool(result)

    def delete(self, key: str) -> int:
        """
        Elimina una clave de Redis.

        Args:
            key: Clave a eliminar

        Returns:
            Número de claves eliminadas
        """
        if not self._client:
            logger.error("Redis client not connected when trying to delete key")
            raise RuntimeError("Redis client not connected")
        return self._client.delete(key)

    def exists(self, key: str) -> bool:
        """
        Verifica si una clave existe en Redis.

        Args:
            key: Clave a verificar

        Returns:
            True si existe, False si no
        """
        if not self._client:
            logger.error("Redis client not connected when trying to check key existence")
            raise RuntimeError("Redis client not connected")
        return bool(self._client.exists(key))

    def expire(self, key: str, seconds: int) -> bool:
        """
        Establece un tiempo de expiración para una clave.

        Args:
            key: Clave a la que establecer expiración
            seconds: Segundos hasta que expire

        Returns:
            True si fue exitoso
        """
        if not self._client:
            logger.error("Redis client not connected when trying to set key expiration")
            raise RuntimeError("Redis client not connected")
        return bool(self._client.expire(key, seconds))


class SessionStore:
    """
    Adapter especializado para almacenar sesiones de chat en Redis.
    Abstrae la serialización y deserialización de mensajes.
    """

    def __init__(self, redis_adapter: RedisAdapter, default_ttl: int = 86400):
        """
        Inicializa el almacén de sesiones.

        Args:
            redis_adapter: Instancia del adaptador de Redis
            default_ttl: Tiempo de vida por defecto en segundos (24 horas)
        """
        self.redis = redis_adapter
        self.default_ttl = default_ttl
        self.key_prefix = "session:"

    def _make_key(self, session_id: str) -> str:
        """
        Genera la clave de Redis para una sesión.

        Args:
            session_id: ID de la sesión

        Returns:
            Clave completa con prefijo
        """
        return f"{self.key_prefix}{session_id}"

    def save_session(self, session_id: str, data: Any, ttl: Optional[int] = None) -> bool:
        """
        Guarda datos de sesión en Redis.

        Args:
            session_id: ID de la sesión
            data: Datos a guardar (serán serializados a JSON)
            ttl: Tiempo de vida en segundos (usa default_ttl si no se especifica)

        Returns:
            True si fue exitoso
        """
        try:
            key = self._make_key(session_id)
            serialized_data = json.dumps(data).encode('utf-8')
            expire_time = ttl if ttl is not None else self.default_ttl

            success = self.redis.set(key, serialized_data, expire_seconds=expire_time)

            return success
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
            return False

    def load_session(self, session_id: str) -> Optional[Any]:
        """
        Carga datos de sesión desde Redis.

        Args:
            session_id: ID de la sesión

        Returns:
            Datos deserializados o None si no existe
        """
        try:
            key = self._make_key(session_id)
            data = self.redis.get(key)

            if data is None:
                return None

            deserialized_data = json.loads(data.decode('utf-8'))

            return deserialized_data
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """
        Elimina una sesión de Redis.

        Args:
            session_id: ID de la sesión

        Returns:
            True si fue eliminada
        """
        try:
            key = self._make_key(session_id)
            deleted = self.redis.delete(key)

            return deleted > 0
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

    def session_exists(self, session_id: str) -> bool:
        """
        Verifica si una sesión existe en Redis.

        Args:
            session_id: ID de la sesión

        Returns:
            True si existe
        """
        key = self._make_key(session_id)
        return self.redis.exists(key)

    def refresh_session(self, session_id: str, ttl: Optional[int] = None) -> bool:
        """
        Renueva el tiempo de expiración de una sesión.

        Args:
            session_id: ID de la sesión
            ttl: Nuevo tiempo de vida en segundos (usa default_ttl si no se especifica)

        Returns:
            True si fue exitoso
        """
        try:
            key = self._make_key(session_id)
            expire_time = ttl if ttl is not None else self.default_ttl

            success = self.redis.expire(key, expire_time)

            return success
        except Exception as e:
            logger.error(f"Error refreshing session {session_id}: {e}")
            return False


def create_redis_adapter() -> RedisAdapter:
    """
    Factory function para crear un adaptador de Redis con la configuración del entorno.

    Returns:
        Instancia configurada y conectada de RedisAdapter
    """
    redis_host = env.get('REDIS_HOST', 'localhost')
    redis_port = int(env.get('REDIS_PORT', '6379'))
    redis_db = int(env.get('REDIS_DB', '0'))

    adapter = RedisAdapter(host=redis_host, port=redis_port, db=redis_db)
    adapter.connect()

    return adapter


def create_session_store(redis_adapter: Optional[RedisAdapter] = None) -> SessionStore:
    """
    Factory function para crear un almacén de sesiones.

    Args:
        redis_adapter: Adaptador de Redis (crea uno nuevo si no se proporciona)

    Returns:
        Instancia configurada de SessionStore
    """
    if redis_adapter is None:
        redis_adapter = create_redis_adapter()

    return SessionStore(redis_adapter)
