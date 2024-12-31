# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock

import pytest
from helpers.util import FakeData

from roc.event import Event, EventBus


class TestEventBus:
    def test_eventbus_send(self, eb_reset, mocker, fake_component):
        eb = EventBus[FakeData]("test")
        stub: MagicMock = mocker.stub(name="event_callback")
        eb.subject.subscribe(stub)

        eb_conn = eb.connect(fake_component)
        d = FakeData("bar", 42)
        eb_conn.send(d)

        stub.assert_called_once()
        assert isinstance(stub.call_args.args[0], Event)
        e = stub.call_args.args[0]
        assert isinstance(e.data, FakeData)
        assert e.data.foo == "bar"
        assert e.data.baz == 42

    def test_eventbus_multiple_listeners(self, eb_reset, mocker, fake_component):
        eb = EventBus[FakeData]("test")
        l1: MagicMock = mocker.stub(name="listener1")
        l2: MagicMock = mocker.stub(name="listener2")
        l3: MagicMock = mocker.stub(name="listener3")
        eb.subject.subscribe(l1)
        eb.subject.subscribe(l2)
        eb.subject.subscribe(l3)

        eb_conn = eb.connect(fake_component)
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

    def test_event_conn_cache(self, eb_reset, mocker, fake_component):
        eb = EventBus[FakeData]("test")
        stub: MagicMock = mocker.stub(name="event_callback")
        eb.subject.subscribe(stub)

        eb_conn = eb.connect(fake_component, cache_depth=10)
        assert eb_conn.cache is not None
        assert len(eb_conn.cache) == 0

        d = FakeData("bar", 42)
        eb_conn.send(d)

        assert len(eb_conn.cache) == 1
        assert eb_conn.cache[0].data is d

    def test_event_conn_cache_multi(self, eb_reset, mocker, fake_component):
        eb = EventBus[FakeData]("test")
        stub: MagicMock = mocker.stub(name="event_callback")
        eb.subject.subscribe(stub)

        eb_conn = eb.connect(fake_component, cache_depth=10)
        assert eb_conn.cache is not None
        assert len(eb_conn.cache) == 0

        d1 = FakeData("bar", 41)
        d2 = FakeData("bar", 42)
        d3 = FakeData("bar", 43)
        eb_conn.send(d1)
        eb_conn.send(d2)
        eb_conn.send(d3)

        assert len(eb_conn.cache) == 3
        assert eb_conn.cache[0].data is d1
        assert eb_conn.cache[1].data is d2
        assert eb_conn.cache[2].data is d3

    def test_event_conn_cache_one(self, eb_reset, mocker, fake_component):
        eb = EventBus[FakeData]("test")
        stub: MagicMock = mocker.stub(name="event_callback")
        eb.subject.subscribe(stub)

        eb_conn = eb.connect(fake_component, cache_depth=1)
        assert eb_conn.cache is not None
        assert len(eb_conn.cache) == 0

        d1 = FakeData("bar", 41)
        d2 = FakeData("bar", 42)
        d3 = FakeData("bar", 43)
        eb_conn.send(d1)
        eb_conn.send(d2)
        eb_conn.send(d3)

        assert len(eb_conn.cache) == 1
        assert eb_conn.cache[0].data is d3
