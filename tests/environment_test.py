# mypy: disable-error-code="no-untyped-def"
from roc.environment import EnvData


class TestEnvInput:
    def test_env_input(self, env_bus_conn) -> None:
        e = EnvData()
        env_bus_conn.send(e)
