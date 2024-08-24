"""This module contains all the settings for the system."""

from __future__ import annotations

import sys
import warnings
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigInitWarning(Warning):
    """A Warning for when attempting to access config before it has been initialized."""

    pass


_config_singleton: Config | None = None

config_settings = {
    "env_prefix": "roc_",
    "env_file": ".env",
}

# XXX: this is hacky to include in this module rather than in the tests, but by
# the time pytest is running this is already set... the purpose of this code is
# to make sure that .env and roc_* environment variables don't alter the
# behavior of tests
if "pytest" in sys.modules:
    config_settings["env_prefix"] = (
        "somereallyweirdrandomstringsothattestingdoesntpickupextraneousstuff"
    )
    config_settings["env_file"] = (
        "somereallyweirdrandomstringsothattestingdoesntpickupextraneousstuff"
    )


class Config(BaseSettings):
    """A Pydantic settings model for configuration of the agent."""

    global config_settings
    model_config = SettingsConfigDict(
        # XXX: can't do **config_settings 'cause of TypedDict?
        env_prefix=config_settings["env_prefix"],
        env_file=config_settings["env_file"],
        extra="forbid",
    )
    db_host: str = Field(default="127.0.0.1")
    db_port: int = Field(default=7687)
    db_conn_encrypted: bool = Field(default=False)
    db_username: str = Field(default="")
    db_password: str = Field(default="")
    db_lazy: bool = Field(default=False)
    node_cache_size: int = Field(default=2**30)
    edge_cache_size: int = Field(default=2**30)
    log_enable: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_modules: str = Field(default="")
    default_action: str = Field(default="pass")
    action_count: int | None = None  # configured by the gymnasium
    observation_shape: tuple[int, ...] | None = None  # configured by the gymnasium
    perception_components: list[str] = Field(
        default=[
            "delta:perception",
            "distance:perception",
            "flood:perception",
            "motion:perception",
            "single:perception",
            "line:perception",
            "color:perception",
            "shape:perception",
        ]
    )
    enable_gym_dump_env: bool = Field(default=False)
    max_dump_frames: int = Field(default=10)

    @staticmethod
    def get() -> Config:
        """Returns the config singleton, which is strongly typed and can be used to
        get or set configuration settings.

        Returns:
            Config: The configuration for ROC.
        """
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
        """Reset the configuration. Mostly used for testing."""
        global _config_singleton
        _config_singleton = None
