from typing import Any

from dynaconf import Dynaconf, Validator


class DefaultSetting(Validator):
    def __init__(self, name: str, val: Any, *, must_exist: bool = True):
        super().__init__(name, default=val, apply_default_on_none=True, must_exist=must_exist)


settings = Dynaconf(
    envvar_prefix="ROC",
    settings_files=["settings.toml", ".secrets.toml"],
    validators=[
        DefaultSetting("db_host", "127.0.0.1"),
        DefaultSetting("db_port", 7687),
        DefaultSetting("log_level", "trace"),
    ],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
