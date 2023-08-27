from roc.config import get_setting


class TestConfig:
    def test_config(self) -> None:
        assert get_setting("db_host", str) == "127.0.0.1"

    def test_config_db_port_default(self) -> None:
        assert get_setting("db_port", int) == 7687
