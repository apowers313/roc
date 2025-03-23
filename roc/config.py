"""This module contains all the settings for the system."""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigInitWarning(Warning):
    """A Warning for when attempting to access config before it has been
    initialized or attempting to init after init has already been performed.
    """


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


# intrinsics configs
class ConfigIntrinsicBase(BaseModel):
    type: str
    name: str
    config: Any


class ConfigPercentIntrinsic(ConfigIntrinsicBase):
    type: Literal["percent"] = "percent"
    config: str


class ConfigMapIntrinsic(ConfigIntrinsicBase):
    type: Literal["map"] = "map"
    config: dict[int, float]


class ConfigIntIntrinsic(ConfigIntrinsicBase):
    type: Literal["int"] = "int"
    config: tuple[int, int]


class ConfigBoolIntrinsic(ConfigIntrinsicBase):
    type: Literal["bool"] = "bool"
    config: None = None


ConfigIntrinsicType = Annotated[
    ConfigPercentIntrinsic | ConfigMapIntrinsic | ConfigIntIntrinsic | ConfigBoolIntrinsic,
    Field(discriminator="type"),
]


class Config(BaseSettings):
    """A Pydantic settings model for configuration of the agent."""

    global config_settings
    model_config = SettingsConfigDict(
        # XXX: can't do **config_settings 'cause of TypedDict?
        env_prefix=config_settings["env_prefix"],
        env_file=config_settings["env_file"],
        extra="forbid",
    )
    # database config
    db_host: str = Field(default="127.0.0.1")
    db_port: int = Field(default=7687)
    db_conn_encrypted: bool = Field(default=False)
    db_username: str = Field(default="")
    db_password: str = Field(default="")
    db_lazy: bool = Field(default=False)
    db_strict_schema: bool = Field(default=True)
    db_strict_schema_warns: bool = Field(default=False)
    # graph config
    node_cache_size: int = Field(default=2**30)
    edge_cache_size: int = Field(default=2**30)
    # log config
    log_enable: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_modules: str = Field(default="")
    # observability config
    observability_logging: bool = Field(default=True)
    observability_logging_level: str = Field(default="DEBUG")
    observability_metrics: bool = Field(default=True)
    observability_metrics_interval: int = Field(default=5000)
    observability_tracing: bool = Field(default=True)
    observability_profiling: bool = Field(default=True)
    observability_host: str = Field(default="http://hal.ato.ms:4317")
    observability_profiling_host: str = Field(default="http://hal.ato.ms:4040")
    # agent config
    gym_actions: tuple[int, ...] | None = Field(default=None)  # configured by the gym
    observation_shape: tuple[int, ...] | None = Field(default=None)  # configured by the gym
    allow_unknown_intrinsic: bool = Field(default=True)
    # jupyter config
    status_update: int = Field(default=50)
    experiment_dir: str = Field(default="/home/apowers/experiment")
    data_dir: str = Field(default="/home/apowers/data")
    # gym config
    num_games: int = Field(default=5)
    enable_gym_dump_env: bool = Field(default=False)
    dump_file: str = Field(default=f"env_dump-{datetime.now().strftime('%Y.%m.%d-%H.%M.%S')}.py")
    max_dump_frames: int = Field(default=10)
    # nethack config
    nethack_extra_options: list[str] = ["autoopen"]
    nethack_max_turns: int = Field(default=100000)
    # experiment modules
    expmod_dirs: list[str] = ["experiments/modules"]
    expmods: list[str] = []
    expmods_use: list[str] = ["action:weighted"]
    # component config
    perception_components: list[tuple[str, str]] = Field(
        default=[
            ("delta", "perception"),
            ("distance", "perception"),
            ("flood", "perception"),
            ("motion", "perception"),
            ("single", "perception"),
            ("line", "perception"),
            ("color", "perception"),
            ("shape", "perception"),
            ("phoneme", "perception"),
        ]
    )
    # intrinsic config
    intrinsics: list[ConfigIntrinsicType] = Field(
        default=[
            ConfigPercentIntrinsic(name="hp", config="hpmax"),
            ConfigPercentIntrinsic(name="ene", config="enemax"),
            ConfigMapIntrinsic(
                name="hunger",
                config={
                    0: 0.5,  # SATIATED = 0
                    1: 1.0,  # NOT_HUNGRY = 1
                    2: 0.75,  # HUNGRY = 2
                    3: 0.5,  # WEAK = 3
                    4: 0.25,  # FAINTING = 4
                    5: 0.1,  # FAINTED = 5
                    6: 0.0,  # STARVED = 6
                },
            ),
            # char
            # wis
            # intel
            # con
            # dex
            # str
        ],
    )
    # significance config
    significance_weights: dict[str, float] = Field(
        default={
            "hp": 10.0,
        }
    )

    def __str__(self) -> str:
        ret = ""
        d = self.dict()
        for k in d:
            ret += f"{k} = {str(d[k])}\n"

        return ret

    @staticmethod
    def print() -> None:
        print(Config.get())  # noqa: T201

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


Config.init()
