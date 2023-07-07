from roc.component import Component
from roc.event import EventBus


class TestData:
    def __init__(self, foo: str, baz: int):
        self.foo = foo
        self.baz = baz


def test_eventbus_send():
    eb = EventBus[TestData]()
    c = Component("test_component", "test_type")
    eb_conn = eb.connect(c)
    d = TestData("bar", 42)
    eb_conn.send(d)
