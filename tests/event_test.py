import pytest

from roc.component import Component
from roc.event import EventBus


class FakeData:
    def __init__(self, foo: str, baz: int):
        self.foo = foo
        self.baz = baz


class TestEventBus:
    def test_eventbus_send(self):
        eb = EventBus[FakeData]("test")
        c = Component("test_component", "test_type")
        eb_conn = eb.connect(c)
        d = FakeData("bar", 42)
        eb_conn.send(d)

    def test_eventbus_duplicate_name(self):
        EventBus.clear_names()
        eb1 = EventBus[FakeData]("test")
        with pytest.raises(Exception, match="Duplicate EventBus name: test"):
            eb2 = EventBus[FakeData]("test")
