from roc.config import settings


class TestConfig:
    def test_config(self) -> None:
        assert settings.db_host == "127.0.0.1"

    def test_config_db_port_default(self) -> None:
        assert settings.db_port == 7687
