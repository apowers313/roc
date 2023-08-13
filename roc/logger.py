import sys

from loguru import logger

from .config import settings

if settings.log_enable:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    # TODO: .add(filter=filter_fn)
    # filter_fn(record) -> bool: if record.module = "x" return False

# https://loguru.readthedocs.io/en/stable/api/logger.html
