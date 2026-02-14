"""
Utilidades para lectura y escritura de archivos en disco.
"""
import os
import base64
import uuid
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)


def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Lee un archivo de texto y retorna su contenido como string.

    Args:
        file_path: Ruta al archivo a leer
        encoding: Codificación del archivo (default: 'utf-8')

    Returns:
        Contenido del archivo como string

    Raises:
        FileNotFoundError: Si el archivo no existe
        UnicodeDecodeError: Si hay problemas con la codificación
        IOError: Si hay problemas al leer el archivo
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"El archivo no existe: {file_path}")

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        return content
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {file_path} with {encoding}: {e}")
        raise
    except IOError as e:
        logger.error(f"IO error reading {file_path}: {e}")
        raise


def read_text_file_safe(file_path: str, encoding: str = 'utf-8', default: Optional[str] = None) -> Optional[str]:
    """
    Lee un archivo de texto de forma segura, retornando un valor por defecto en caso de error.

    Args:
        file_path: Ruta al archivo a leer
        encoding: Codificación del archivo (default: 'utf-8')
        default: Valor a retornar en caso de error (default: None)

    Returns:
        Contenido del archivo como string, o el valor por defecto si hay error
    """
    try:
        return read_text_file(file_path, encoding)
    except Exception as e:
        logger.warning(f"Could not read file {file_path}: {e}. Returning default value.")
        return default


def save_image_file(image_data: bytes, storage_dir: str, prefix: str, extension: str) -> str:
    """
    Guarda una imagen desde bytes y retorna la ruta relativa del archivo.

    Args:
        image_data: Bytes de la imagen
        storage_dir: Directorio donde guardar la imagen
        prefix: Prefijo para el nombre del archivo
        extension: Extensión del archivo

    Returns:
        Ruta relativa del archivo guardado

    Raises:
        ValueError: Si los datos de imagen son inválidos
        IOError: Si hay problemas al guardar el archivo
    """
    try:
        if not image_data:
            raise ValueError("No se proporcionaron datos de imagen")

        os.makedirs(storage_dir, exist_ok=True)

        filename = f"{prefix}_{uuid.uuid4().hex}.{extension}"
        file_path = os.path.join(storage_dir, filename)

        with open(file_path, 'wb') as f:
            f.write(image_data)

        return filename

    except IOError as e:
        logger.error(f"IO error saving image: {e}")
        raise


def delete_file(file_path: str, storage_dir: str) -> bool:
    """
    Elimina un archivo de forma segura.

    Args:
        file_path: Ruta relativa del archivo a eliminar (nombre del archivo)
        storage_dir: Directorio donde se encuentra el archivo

    Returns:
        True si el archivo fue eliminado, False si no existía
    """
    try:
        full_path = os.path.join(storage_dir, file_path)
        if os.path.exists(full_path):
            os.remove(full_path)
            return True
        else:
            logger.warning(f"File not found for deletion: {full_path}")
            return False
    except Exception as e:
        logger.error(f"Error deleting file {full_path}: {e}")
        raise
