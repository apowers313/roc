# from typing import Generic, TypeVar

# from dynaconf import Dynaconf, Validator

from roc.config import Config

# ValType = TypeVar("ValType")


# class DefaultSetting(Validator, Generic[ValType]):
#     def __init__(
#         self, name: str, cast_type: type[ValType], val: ValType, *, must_exist: bool = True
#     ) -> None:
#         super().__init__(
#             name,
#             default=val,
#             apply_default_on_none=True,
#             must_exist=must_exist,
#             cast=str,
#         )


# settings_vars = [
#     DefaultSetting("foo", "str", "this is foo"),
# ]

# blah = Dynaconf(
#     envvar_prefix="ROC",
#     settings_files=["settings.toml", ".secrets.toml"],
#     validators=settings_vars,
# )


class TestConfig:
    # def test_foo(self) -> None:
    #     s = blah.foo
    #     print("blah.foo", s)
    #     print("is str", isinstance(s, str))
    #     reveal_type(s)

    def test_config(self) -> None:
        settings = Config.get()
        assert settings.DB_HOST == "127.0.0.1"

    def test_config_db_port_default(self) -> None:
        settings = Config.get()
        assert settings.DB_PORT == 7687

    # def test_config_default_components(self) -> None:
    #     s = get_setting("default_components", list)

    #     assert isinstance(s, list)
    #     assert "action" in s
