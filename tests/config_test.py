# mypy: disable-error-code="no-untyped-def"

from roc.config import Config


class TestConfig:
    def test_config(self) -> None:
        settings = Config.get()
        assert settings.db_host == "127.0.0.1"

    def test_config_db_port_default(self) -> None:
        settings = Config.get()
        assert settings.db_port == 7687

    def test_passed_config(self) -> None:
        Config.reset()
        Config.init({"db_host": "x.y.z"})
        settings = Config.get()
        assert settings.db_host == "x.y.z"
