from unittest.mock import MagicMock

import pytest

from roc.component import Component
from roc.event import Event, EventBus


class FakeData:
    def __init__(self, foo: str, baz: int):
        self.foo = foo
        self.baz = baz


class TestEventBus:
    def test_eventbus_send(self, mocker):
        eb = EventBus[FakeData]("test")
        stub: MagicMock = mocker.stub(name="event_callback")
        eb.subject.subscribe(stub)

        c = Component("test_component", "test_type")
        eb_conn = eb.connect(c)
        d = FakeData("bar", 42)
        eb_conn.send(d)

        stub.assert_called_once()
        assert isinstance(stub.call_args.args[0], Event)
        e = stub.call_args.args[0]
        assert isinstance(e.data, FakeData)
        assert e.data.foo == "bar"
        assert e.data.baz == 42

    def test_eventbus_duplicate_name(self):
        EventBus.clear_names()
        eb1 = EventBus[FakeData]("test")
        with pytest.raises(Exception, match="Duplicate EventBus name: test"):
            eb2 = EventBus[FakeData]("test")
