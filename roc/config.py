"""This module contains all the settings for the system."""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigInitWarning(Warning):
    """A Warning for when attempting to access config before it has been initialized."""

    pass


_config_singleton: Config | None = None


class Config(BaseSettings):
    """A Pydantic settings model for configuration of the agent."""

    model_config = SettingsConfigDict(
        env_prefix="roc_",
        env_file=".env",
        extra="forbid",
    )
    db_host: str = Field(default="127.0.0.1")
    db_port: int = Field(default=7687)
    db_conn_encrypted: bool = Field(default=False)
    db_username: str = Field(default="")
    db_password: str = Field(default="")
    db_lazy: bool = Field(default=False)
    node_cache_size: int = Field(default=2**16)
    edge_cache_size: int = Field(default=2**16)
    # node_cache_size: int = Field(default=2**11)
    # edge_cache_size: int = Field(default=2**11)
    log_enable: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_modules: str = Field(default="")
    default_action: str = Field(default="pass")
    perception_components: list[str] = Field(default=["delta:perception"])
    enable_gym_dump_env: bool = Field(default=False)
    max_dump_frames: int = Field(default=10)
    nethack_spectrum: str = Field(pattern=r"chars|colors|glyphs", default="chars")

    @staticmethod
    def get() -> Config:
        global _config_singleton
        if _config_singleton is None:
            warnings.warn(
                "Getting settings before config module was initialized. Please call init() first",
                ConfigInitWarning,
            )
            Config.init()
            assert _config_singleton is not None
        return _config_singleton

    @staticmethod
    def init(
        config: dict[str, Any] | None = None,
        *,
        force: bool = False,
        use_secrets: bool = True,
    ) -> None:
        """Initializes the settings by reading the configuration files and environment variables"""

        global _config_singleton
        initialized = _config_singleton is not None
        if initialized and not force:
            warnings.warn(
                "Config already initialized, returning existing configuration.",
                ConfigInitWarning,
            )
            return

        passed_conf = config or {}
        _config_singleton = Config(**passed_conf)

    @staticmethod
    def reset() -> None:
        global _config_singleton
        _config_singleton = None
