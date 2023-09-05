from __future__ import annotations

import warnings
from typing import Any

from dynaconf import Dynaconf
from pydantic import BaseModel, Field, model_validator


class LogImportWarning(Warning):
    pass


_config_singleton: Config | None = None


class CaseInsensitiveModel(BaseModel):
    @model_validator(mode="before")
    def __lowercase_property_keys__(cls, values: Any) -> Any:
        def __lower__(value: Any) -> Any:
            if isinstance(value, dict):
                return {k.lower(): __lower__(v) for k, v in value.items()}
            return value

        return __lower__(values)


# XXX: no promises that these are complete or correct...
class DynaconfConfig(BaseModel):
    ENVVAR_PREFIX_FOR_DYNACONF: str | None
    SETTINGS_FILE_FOR_DYNACONF: list[str]
    RENAMED_VARS: dict[str, Any]  # TODO
    ROOT_PATH_FOR_DYNACONF: str | None
    ENVIRONMENTS_FOR_DYNACONF: bool
    MAIN_ENV_FOR_DYNACONF: str
    LOWERCASE_READ_FOR_DYNACONF: bool
    ENV_SWITCHER_FOR_DYNACONF: str
    FORCE_ENV_FOR_DYNACONF: str | None
    DEFAULT_ENV_FOR_DYNACONF: str
    IGNORE_UNKNOWN_ENVVARS_FOR_DYNACONF: bool
    AUTO_CAST_FOR_DYNACONF: bool
    ENCODING_FOR_DYNACONF: str
    MERGE_ENABLED_FOR_DYNACONF: bool
    DOTTED_LOOKUP_FOR_DYNACONF: bool
    NESTED_SEPARATOR_FOR_DYNACONF: str | None
    ENVVAR_FOR_DYNACONF: str | None
    REDIS_FOR_DYNACONF: dict[str, Any]  # TODO
    REDIS_ENABLED_FOR_DYNACONF: bool
    VAULT_FOR_DYNACONF: dict[str, Any]  # TODO
    VAULT_ENABLED_FOR_DYNACONF: bool
    VAULT_PATH_FOR_DYNACONF: str | None
    VAULT_MOUNT_POINT_FOR_DYNACONF: str | None
    VAULT_ROOT_TOKEN_FOR_DYNACONF: str | None
    VAULT_KV_VERSION_FOR_DYNACONF: int
    VAULT_AUTH_WITH_IAM_FOR_DYNACONF: bool
    VAULT_AUTH_ROLE_FOR_DYNACONF: str | None
    VAULT_ROLE_ID_FOR_DYNACONF: str | None
    VAULT_SECRET_ID_FOR_DYNACONF: str | None
    VAULT_USERNAME_FOR_DYNACONF: str | None
    VAULT_PASSWORD_FOR_DYNACONF: str | None
    CORE_LOADERS_FOR_DYNACONF: list[str]
    LOADERS_FOR_DYNACONF: list[str]
    SILENT_ERRORS_FOR_DYNACONF: bool
    FRESH_VARS_FOR_DYNACONF: list[str]
    DOTENV_PATH_FOR_DYNACONF: str | None
    DOTENV_VERBOSE_FOR_DYNACONF: bool
    DOTENV_OVERRIDE_FOR_DYNACONF: bool
    INSTANCE_FOR_DYNACONF: str | None
    YAML_LOADER_FOR_DYNACONF: str
    COMMENTJSON_ENABLED_FOR_DYNACONF: bool
    SECRETS_FOR_DYNACONF: str | None
    INCLUDES_FOR_DYNACONF: list[str]
    PRELOAD_FOR_DYNACONF: list[str]
    SKIP_FILES_FOR_DYNACONF: list[str]
    APPLY_DEFAULT_ON_NONE_FOR_DYNACONF: None
    VALIDATE_ON_UPDATE_FOR_DYNACONF: bool
    SYSENV_FALLBACK_FOR_DYNACONF: bool
    DYNACONF_NAMESPACE: str
    NAMESPACE_FOR_DYNACONF: str
    DYNACONF_SETTINGS_MODULE: list[str]
    DYNACONF_SETTINGS: list[str]
    SETTINGS_MODULE: list[str]
    SETTINGS_MODULE_FOR_DYNACONF: list[str]
    PROJECT_ROOT: str | None
    PROJECT_ROOT_FOR_DYNACONF: str | None
    DYNACONF_SILENT_ERRORS: bool
    DYNACONF_ALWAYS_FRESH_VARS: list[str]
    BASE_NAMESPACE_FOR_DYNACONF: str
    GLOBAL_ENV_FOR_DYNACONF: str
    ENV_FOR_DYNACONF: str | None


# XXX: all attributes must be uppercase because Dynaconf converts everything to uppercase
# might be optional in Dynaconf 4.0:
# https://github.com/dynaconf/dynaconf/issues/761
class Config(DynaconfConfig, extra="forbid", validate_default=True):
    DB_HOST: str = Field(default="127.0.0.1")
    DB_PORT: int = Field(default=7687)
    DB_CONN_ENCRYPTED: bool = Field(default=False)
    DB_USERNAME: str = Field(default="")
    DB_PASSWORD: str = Field(default="")
    DB_LAZY: bool = Field(default=False)
    NODE_CACHE_SIZE: int = Field(default=2**11)
    EDGE_CACHE_SIZE: int = Field(default=2**11)
    LOG_ENABLE: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")
    LOG_MODULES: str = Field(default="")
    DEFAULT_ACTION: str = Field(default="pass")
    PERCEPTION_COMPONENTS: list[str] = Field(default=["delta:perception"])

    @staticmethod
    def get() -> Config:
        global _config_singleton
        if _config_singleton is None:
            warnings.warn(
                "Getting settings before config module was initialized. Please call init() first",
                LogImportWarning,
            )
            Config.init()
            assert _config_singleton is not None
        return _config_singleton

    @staticmethod
    def init(*, force: bool = False, use_secrets: bool = True) -> None:
        """Initializes the settings by reading the configuration files and environment variables"""

        global _config_singleton
        initialized = _config_singleton is not None
        if initialized and not force:
            return

        settings_files = ["settings.toml"]
        if use_secrets:
            settings_files.append(".secrets.toml")

        dynaconf_settings = Dynaconf(
            envvar_prefix="ROC",
            settings_files=settings_files,
        )

        _config_singleton = Config(**dynaconf_settings)
