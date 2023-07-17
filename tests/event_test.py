from unittest.mock import MagicMock

import pytest

from roc.component import Component
from roc.event import Event, EventBus


class FakeData:
    def __init__(self, foo: str, baz: int):
        self.foo = foo
        self.baz = baz


class TestEventBus:
    def test_eventbus_send(self, eb_reset, mocker):
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

    def test_eventbus_multiple_listeners(self, eb_reset, mocker):
        eb = EventBus[FakeData]("test")
        l1: MagicMock = mocker.stub(name="listener1")
        l2: MagicMock = mocker.stub(name="listener2")
        l3: MagicMock = mocker.stub(name="listener3")
        eb.subject.subscribe(l1)
        eb.subject.subscribe(l2)
        eb.subject.subscribe(l3)

        c = Component("test_component", "test_type")
        eb_conn = eb.connect(c)
        d = FakeData("bar", 42)
        eb_conn.send(d)

        assert l1.call_count == 1
        e1 = l1.call_args.args[0]
        assert id(e1.data) == id(d)

        assert l2.call_count == 1
        e2 = l2.call_args.args[0]
        assert id(e2.data) == id(d)

        assert l3.call_count == 1
        e3 = l3.call_args.args[0]
        assert id(e3.data) == id(d)

    def test_eventbus_duplicate_name(self, eb_reset):
        EventBus.clear_names()
        EventBus[FakeData]("test")
        with pytest.raises(Exception, match="Duplicate EventBus name: test"):
            EventBus[FakeData]("test")
