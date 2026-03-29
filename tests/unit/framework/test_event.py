# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/event.py."""

from unittest.mock import MagicMock

import pytest

from roc.component import ComponentId
from roc.event import BusConnection, Event, EventBus, eventbus_names


@pytest.fixture(autouse=True)
def clear_eventbus_names():
    """Clear eventbus names before/after each test."""
    saved = eventbus_names.copy()
    eventbus_names.clear()
    yield
    eventbus_names.clear()
    eventbus_names.update(saved)


def _make_mock_component(name="test", type="test_type"):
    """Create a mock component with the minimum interface needed."""
    comp = MagicMock()
    comp.name = name
    comp.type = type
    comp.id = ComponentId(type, name)
    comp.event_filter = MagicMock(return_value=True)
    return comp


class TestEventBus:
    def test_creation_with_name(self):
        bus = EventBus[int]("test_bus")
        assert bus.name == "test_bus"
        assert "test_bus" in eventbus_names

    def test_duplicate_name_raises(self):
        EventBus[int]("dup_bus")
        with pytest.raises(Exception, match="Duplicate EventBus name"):
            EventBus[int]("dup_bus")

    def test_clear_names(self):
        EventBus[int]("bus_a")
        EventBus[int]("bus_b")
        assert len(eventbus_names) == 2
        EventBus.clear_names()
        assert len(eventbus_names) == 0

    def test_cache_depth_zero(self):
        bus = EventBus[int]("no_cache")
        assert bus.cache_depth == 0
        assert bus.cache is None

    def test_cache_depth_positive(self):
        bus = EventBus[int]("cached", cache_depth=3)
        assert bus.cache_depth == 3
        assert bus.cache is not None
        assert len(bus.cache) == 0

    def test_connect_returns_bus_connection(self):
        bus = EventBus[int]("conn_bus")
        comp = _make_mock_component()
        conn = bus.connect(comp)
        assert isinstance(conn, BusConnection)
        assert conn.attached_bus is bus
        assert conn.attached_component is comp


class TestBusConnection:
    def test_send_creates_event_and_pushes(self):
        bus = EventBus[int]("send_bus")
        comp = _make_mock_component()
        conn = bus.connect(comp)

        received = []
        bus.subject.subscribe(lambda e: received.append(e))
        conn.send(42)
        assert len(received) == 1
        assert received[0].data == 42
        assert received[0].src_id == comp.id

    def test_close_disposes_subscribers(self):
        bus = EventBus[int]("close_bus")
        comp = _make_mock_component()
        conn = bus.connect(comp)
        comp.event_filter.return_value = True
        conn.listen(lambda e: None)
        assert len(conn.subscribers) == 1
        conn.close()
        # After close, subscribers should be disposed
        assert all(sub.is_disposed for sub in conn.subscribers)  # type: ignore[attr-defined]


class TestEvent:
    def test_repr_formatting(self):
        bus = EventBus[str]("repr_bus")
        comp_id = ComponentId("mytype", "myname")
        e = Event[str]("hello", comp_id, bus)
        r = repr(e)
        assert "myname:mytype" in r
        assert "repr_bus" in r
        assert "hello" in r

    def test_event_stores_data_and_source(self):
        bus = EventBus[int]("data_bus")
        comp_id = ComponentId("t", "n")
        e = Event[int](99, comp_id, bus)
        assert e.data == 99
        assert e.src_id == comp_id
        assert e.bus is bus


class TestCacheDepthIntegration:
    def test_cache_stores_events(self):
        bus = EventBus[int]("cache_int_bus", cache_depth=2)
        comp = _make_mock_component()
        conn = bus.connect(comp)
        conn.send(1)
        conn.send(2)
        assert bus.cache is not None
        assert len(bus.cache) == 2
        assert bus.cache[0].data == 1
        assert bus.cache[1].data == 2

    def test_cache_evicts_old_events(self):
        bus = EventBus[int]("cache_evict_bus", cache_depth=2)
        comp = _make_mock_component()
        conn = bus.connect(comp)
        conn.send(1)
        conn.send(2)
        conn.send(3)
        assert bus.cache is not None
        assert len(bus.cache) == 2
        assert bus.cache[0].data == 2
        assert bus.cache[1].data == 3
