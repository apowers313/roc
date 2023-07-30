from typing import Generic, TypeVar

from dynaconf import Dynaconf, Validator

ValType = TypeVar("ValType")


class DefaultSetting(Validator, Generic[ValType]):
    def __init__(self, name: str, val: ValType, *, must_exist: bool = True):
        super().__init__(name, default=val, apply_default_on_none=True, must_exist=must_exist)


settings = Dynaconf(
    envvar_prefix="ROC",
    settings_files=["settings.toml", ".secrets.toml"],
    validators=[
        DefaultSetting[str]("db_host", "127.0.0.1"),
        DefaultSetting[int]("db_port", 7687),
        DefaultSetting[bool]("db_conn_encrypted", False),
        DefaultSetting[str | None]("db_username", None),
        DefaultSetting[str | None]("db_password", None),
        DefaultSetting[bool]("db_lazy", False),
        DefaultSetting[int]("node_cache_size", 2**11),
        DefaultSetting[int]("edge_cache_size", 2**11),
        DefaultSetting[str]("log_level", "trace"),
    ],
)


# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
