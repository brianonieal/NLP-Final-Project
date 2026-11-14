# GenAI disclosure: Claude (Anthropic) generated this file as part of the
# initial repo skeleton.
"""Logging setup shared by every module in the package.

Modules call ``get_logger(__name__)`` and log through the result. Scripts and
entry points call ``configure_logging()`` once at startup to decide where the
messages go. Library code never configures handlers itself.
"""

import logging
from pathlib import Path

PACKAGE_LOGGER_NAME = "gram_sdpo"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger that sits under the package logger.

    Args:
        name: Usually ``__name__`` of the calling module. Names outside the
            package are nested under it so one call to ``configure_logging``
            controls all of them. ``None`` returns the package logger.

    Returns:
        The requested logger.
    """
    if name is None or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Attach handlers to the package logger.

    Safe to call more than once. Each call replaces the handlers from the
    previous call, so messages are never written twice.

    Args:
        level: Minimum level to emit, as a ``logging`` constant or its name.
        log_file: Optional path. When given, messages also go to this file and
            missing parent folders are created.

    Returns:
        The configured package logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logging configured at level %s", logging.getLevelName(logger.level))
    return logger
