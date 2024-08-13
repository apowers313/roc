"""This module defines the logging interface for storing system logs."""

import pkgutil
import sys
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter, field_validator

from .config import Config

__all__ = ["logger", "init", "default_log_filter"]

module_names = [name for _, name, _ in pkgutil.iter_modules(["roc"])]


class DebugModuleLevel(BaseModel):
    """A Pydantic model for storing the logging level. Used primarily for
    setting per-module logging levels.
    """

    module_name: str
    log_level: str = Field(pattern=r"TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL")

    @field_validator("module_name", mode="before")
    @classmethod
    def validate_module_name(cls, name: str) -> str:
        assert (
            name in module_names
        ), f"Module name '{name}' not a valid module name. Must be one of {module_names}"

        return name


class LogFilter:
    """A Callable filter that determines if the loguru record should be logged
    or not.
    """

    def __init__(
        self,
        *,
        level: str | None = None,
        log_modules: str | None = None,
        enabled: bool = True,
        use_module_settings: bool = True,
    ):
        settings = Config.get()
        self.level = level or settings.log_level
        self.level_num = logger.level(self.level).no
        if not isinstance(log_modules, str):
            if use_module_settings:
                log_modules = settings.log_modules
            else:
                log_modules = ""
        mod_list = self.parse_module_str(log_modules)
        self.module_levels = {mod_lvl.module_name: mod_lvl.log_level for mod_lvl in mod_list}

    def __call__(self, record: Any) -> bool:
        # TODO: this would be more effecient as a dict rather than a loop (O(1) rather than O(n))

        if record["module"] in self.module_levels:
            mod_log_level = self.module_levels[record["module"]]
            mod_level_num = logger.level(mod_log_level).no
            if record["level"].no >= mod_level_num:
                return True
            else:
                return False

        if record["level"].no >= self.level_num:
            return True

        return False

    @classmethod
    def parse_module_str(cls, s: str) -> list[DebugModuleLevel]:
        s = s.strip()

        mod_list = s.split(";")
        # empty str
        if mod_list == [""]:
            return []

        mod_lvl_list: list[dict[str, str]] = []
        for mod in mod_list:
            mod_parts = mod.split(":")
            mod_lvl_list.append({"module_name": mod_parts[0], "log_level": mod_parts[1]})

        debug_module_list = TypeAdapter(list[DebugModuleLevel])
        return debug_module_list.validate_python(mod_lvl_list)


default_log_filter = None


def init() -> None:
    """Initializes the logging module. Installs the filter, fetches the
    settings, etc.
    """
    global default_log_filter
    default_log_filter = LogFilter()

    logger.remove()
    settings = Config.get()
    if settings.log_enable:
        logger.add(sys.stderr, level=0, filter=default_log_filter)
