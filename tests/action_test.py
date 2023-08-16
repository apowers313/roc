# mypy: disable-error-code="no-untyped-def"

from roc.action import ActionComponent, ActionCount


class TestAction:
    def test_action_count(self, action_bus_conn, mocker) -> None:
        ad = ActionCount(action_count=42)
        ac = ActionComponent()
        assert ac.action_count is None
        mocker.spy(ActionComponent, "recv_action_count")
        action_bus_conn.send(ad)

        # spy.assert_called_once()
        # assert spy.call_count == 1
        assert ac.action_count == 42
