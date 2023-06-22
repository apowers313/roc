from roc.config import settings


def test_config():
    assert settings.test == "blargh"
