import pkgutil
import sys
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter, field_validator

from .config import settings

__all__ = [
    "logger",
]

module_names = [name for _, name, _ in pkgutil.iter_modules(["roc"])]


class DebugModuleLevel(BaseModel):
    module_name: str
    log_level: str = Field(pattern=r"TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL")

    @field_validator("module_name", mode="before")
    @classmethod
    def validate_module_name(cls, name: str) -> str:
        assert (
            name in module_names
        ), f"Module name '{name}' not a valid module name. Must be one of {module_names}"

        return name

    # @field_validator("log_level", mode="before")
    # @classmethod
    # def validate_log_level(cls, level: str | int) -> int:
    #     if isinstance(level, int):
    #         return level

    #     return log_to_level(level)


class LogFilter:
    def __init__(
        self, *, level: str | None = None, log_modules: str | None = None, enabled: bool = True
    ):
        self.level = level or settings.log_level
        self.level_num = logger.level(self.level).no
        if isinstance(log_modules, str):
            self.module_levels = self.parse_module_str(log_modules)
        else:
            self.module_levels = []

    def __call__(self, record: Any) -> bool:
        for mod in self.module_levels:
            if record["module"] == mod.module_name:
                mod_level_num = logger.level(mod.log_level).no
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


default_log_filter = LogFilter()

if settings.log_enable:
    logger.remove()
    logger.add(sys.stderr, level=0, filter=default_log_filter)