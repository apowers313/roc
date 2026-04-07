"""This module contains all the settings for the system."""

from __future__ import annotations

import inspect
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
    """Base model for intrinsic configuration entries."""

    type: str
    name: str
    config: Any


class ConfigPercentIntrinsic(ConfigIntrinsicBase):
    """Config for percent-based intrinsics (e.g. hp as a fraction of hpmax)."""

    type: Literal["percent"] = "percent"
    config: str


class ConfigMapIntrinsic(ConfigIntrinsicBase):
    """Config for map-based intrinsics that map discrete values to floats."""

    type: Literal["map"] = "map"
    config: dict[int, float]


class ConfigIntIntrinsic(ConfigIntrinsicBase):
    """Config for integer range intrinsics with a min and max."""

    type: Literal["int"] = "int"
    config: tuple[int, int]


class ConfigBoolIntrinsic(ConfigIntrinsicBase):
    """Config for boolean intrinsics (true/false)."""

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
    ssl_certfile: str | None = Field(default=None)
    ssl_keyfile: str | None = Field(default=None)
    # dashboard config
    dashboard_enabled: bool = Field(default=True)
    dashboard_port: int = Field(default=9042)
    dashboard_callback_url: str | None = Field(default=None)
    dashboard_gzip: bool = Field(default=True)
    emit_state: bool = Field(default=True)
    emit_state_screen: bool = Field(default=True)
    emit_state_saliency: bool = Field(default=True)
    emit_state_features: bool = Field(default=True)
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
    expmods_use: list[tuple[str, str]] = [("action", "weighted")]
    # attention config
    attention_cycles: int = 4
    # saliency-attenuation/linear-decline config
    saliency_attenuation_capacity: int = 5
    saliency_attenuation_radius: int = 3
    saliency_attenuation_max_penalty: float = 1.0
    saliency_attenuation_max_attenuation: float = 0.9
    # saliency-attenuation/active-inference config
    saliency_attenuation_ai_max_states: int = 64
    saliency_attenuation_ai_max_locations: int = 32
    saliency_attenuation_ai_max_attenuation: float = 0.9
    saliency_attenuation_ai_saliency_weight: float = 0.5
    saliency_attenuation_ai_omega_alpha_prior: float = 2.0
    saliency_attenuation_ai_omega_beta_prior: float = 1.0
    saliency_attenuation_ai_zeta_alpha_prior: float = 2.0
    saliency_attenuation_ai_zeta_beta_prior: float = 1.0
    saliency_attenuation_ai_b_self_transition: float = 0.9
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
                    0: 0.5,  # satiated
                    1: 1.0,  # not hungry
                    2: 0.75,  # hungry
                    3: 0.5,  # weak
                    4: 0.25,  # fainting
                    5: 0.1,  # fainted
                    6: 0.0,  # starved
                },
            ),
        ],
    )
    # game threading controls
    use_game_threads: bool = Field(default=False)
    # graphdb controls
    graphdb_export: bool = Field(default=False)
    graphdb_flush: bool = Field(default=False)
    # debugpy controls
    debug_port: int = Field(default=5678)
    debug_wait: bool = Field(default=False)
    # remote logger controls
    debug_remote_log: bool = Field(default=True)
    debug_remote_log_url: str = Field(default="https://dev.ato.ms:9080/log")
    # snapshot controls
    debug_snapshot_interval: int = Field(default=0)  # emit snapshot every N ticks (0 = disabled)
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
        """Prints the current configuration to stdout."""
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
        if _config_singleton is not None and not force:
            _warn_already_initialized(config, _config_singleton)
            return

        passed_conf = config or {}
        _config_singleton = Config(**passed_conf)

    @staticmethod
    def reset() -> None:
        """Reset the configuration. Mostly used for testing."""
        global _config_singleton
        _config_singleton = None


def _warn_already_initialized(config: dict[str, Any] | None, current: Config) -> None:
    """Build and emit a warning when Config.init is called after initialization."""
    frame = inspect.stack()[2]
    caller_info = f"{frame.filename}:{frame.lineno}"
    changed_keys = _find_changed_keys(config or {}, current)

    msg = f"Config already initialized, returning existing configuration. Called from {caller_info}"
    if changed_keys:
        diff = "\n".join(changed_keys)
        msg += f"\nChanged keys that will be IGNORED:\n{diff}"
    warnings.warn(msg, ConfigInitWarning)


def _find_changed_keys(passed_conf: dict[str, Any], current: Config) -> list[str]:
    """Compare passed config against current singleton and return descriptions of differences."""
    changed: list[str] = []
    for key, new_val in passed_conf.items():
        if hasattr(current, key):
            old_val = getattr(current, key)
            if old_val != new_val:
                changed.append(f"  {key}: {old_val!r} -> {new_val!r}")
    return changed


Config.init()
