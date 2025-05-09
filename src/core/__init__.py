from .logger_utils import get_logger

logger = get_logger(__file__)

try:
    from .opik_utils import configure_opik

    configure_opik()
except:
    logger.warning("Could not configure Opik.")

__all__ = ["get_logger", "logger_utils", "db"]
