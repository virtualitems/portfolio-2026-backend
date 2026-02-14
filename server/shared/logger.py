"""
Configuración centralizada de logging para la aplicación.
"""
import logging
from datetime import datetime
from logging import Handler
from pathlib import Path
from typing import Optional

from .env import env


class DailyLogHandler(Handler):
    """Handler que escribe logs en archivos diarios con formato YYYY_MM_DD.log"""

    def __init__(self, logs_dir: Path, level=logging.NOTSET):
        super().__init__(level)
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(exist_ok=True)
        self.current_date = None
        self.current_handler = None

    def emit(self, record):
        """Emite el log al archivo correspondiente según la fecha actual"""
        today = datetime.now().date()

        # Si cambió el día o es la primera vez, crear nuevo handler
        if self.current_date != today:
            if self.current_handler:
                self.current_handler.close()

            self.current_date = today
            log_filename = today.strftime('%Y_%m_%d.log')
            log_filepath = self.logs_dir / log_filename

            self.current_handler = logging.FileHandler(
                log_filepath,
                mode='a',
                encoding='utf-8'
            )
            self.current_handler.setFormatter(self.formatter)

        if self.current_handler:
            self.current_handler.emit(record)

    def close(self):
        """Cierra el handler actual"""
        if self.current_handler:
            self.current_handler.close()
        super().close()


# Crear carpeta de logs si no existe
logs_dir = Path(__file__).parent.parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True)

log_level = getattr(logging, env.get('LOG_LEVEL', 'INFO').upper())

# Handler diario personalizado
daily_handler = DailyLogHandler(logs_dir)
daily_handler.setLevel(log_level)
daily_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        daily_handler
    ],
    force=True
)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Obtiene un logger configurado para la aplicación.

    Args:
        name: Nombre del logger (típicamente __name__ del módulo)

    Returns:
        Logger configurado con el nivel especificado en variables de entorno
    """
    return logging.getLogger(name)
