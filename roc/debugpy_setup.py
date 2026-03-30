"""Setup for debugpy DAP debugging server."""

import sys

from roc.framework.config import Config
from roc.framework.logger import logger


def maybe_start_debugpy() -> None:
    """Start debugpy listener if debug_port is configured.

    When debug_port > 0, imports debugpy and starts listening on that port.
    When debug_port is 0 (default), does nothing and does not import debugpy.
    Skips if debugpy is already active (e.g. process was launched by a DAP debugger).
    """
    settings = Config.get()
    if settings.debug_port > 0:
        if "pydevd" in sys.modules:
            logger.info("debugpy already active (launched by debugger), skipping listen()")
            return

        import debugpy

        try:
            debugpy.listen(("127.0.0.1", settings.debug_port))
        except RuntimeError as exc:
            logger.warning(f"debugpy failed to listen on port {settings.debug_port}: {exc}")
            return
        logger.info(f"debugpy listening on port {settings.debug_port}")

        if settings.debug_wait:
            logger.info("Waiting for debugger to attach...")
            debugpy.wait_for_client()
            logger.info("Debugger attached.")
