from roc.config import Config


class TestConfig:
    def test_config(self) -> None:
        settings = Config.get()
        assert settings.DB_HOST == "127.0.0.1"

    def test_config_db_port_default(self) -> None:
        settings = Config.get()
        assert settings.DB_PORT == 7687
