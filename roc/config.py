import warnings
from typing import Any, Generic, TypeVar

from dynaconf import Dynaconf, Validator

__all__ = [
    "initialized",
    "get_setting",
]


class LogImportWarning(Warning):
    pass


# def __getattr__(key: str) -> Any:
#     if key == "settings":
#         global initialized
#         if not initialized:
#             warnings.warn(
#                 "Getting settings before config module was initialized. Please call init() first",
#                 LogImportWarning,
#             )
#             init()
#         return _settings
#     raise AttributeError(f"module '{__name__}' has no attribute '{key}'")

ValType = TypeVar("ValType")


class DefaultSetting(Validator, Generic[ValType]):
    def __init__(self, name: str, val: ValType, *, must_exist: bool = True) -> None:
        super().__init__(name, default=val, apply_default_on_none=True, must_exist=must_exist)


settings_vars = [
    DefaultSetting[str]("db_host", "127.0.0.1"),
    DefaultSetting[int]("db_port", 7687),
    DefaultSetting[bool]("db_conn_encrypted", False),
    DefaultSetting[str]("db_username", ""),
    DefaultSetting[str]("db_password", ""),
    DefaultSetting[bool]("db_lazy", False),
    DefaultSetting[int]("node_cache_size", 2**11),
    DefaultSetting[int]("edge_cache_size", 2**11),
    DefaultSetting[str]("log_level", "INFO"),
    DefaultSetting[bool]("log_enable", True),
    DefaultSetting[str]("log_modules", ""),
]

_settings: Any = None
initialized = False


def load_config(*, force: bool = False) -> None:
    """Initializes the settings by reading the configuration files and environment variables"""
    global initialized
    if initialized and not force:
        return

    global _settings
    _settings = Dynaconf(
        envvar_prefix="ROC",
        settings_files=["settings.toml", ".secrets.toml"],
        validators=settings_vars,
    )

    initialized = True
    config_modules()


def config_modules() -> None:
    import roc.logger as roc_logger

    roc_logger.config()


SettingType = TypeVar("SettingType", bound=object)


def get_setting(name: str, t: type[SettingType]) -> SettingType:
    global initialized
    if not initialized:
        warnings.warn(
            "Getting config value before settings were loaded. Please call load_config() first",
            LogImportWarning,
        )
        load_config()

    val: SettingType = _settings[name]
    if not isinstance(val, t):
        raise TypeError(f"Config '{name}' value was not of the specified type {t}")

    return val
