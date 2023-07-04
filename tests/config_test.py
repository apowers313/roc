from roc.config import settings


def test_config():
    assert settings.db_host == "127.0.0.1"


def test_config_db_port_default():
    assert settings.db_port == 7687
