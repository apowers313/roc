from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="ROC",
    settings_files=["settings.toml", ".secrets.toml"],
    validators=[Validator("db_host", must_exist=True), Validator("db_port", default=7687, apply_default_on_none=True)],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
