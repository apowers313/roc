from roc.eventbus import EventBus


def test_get_bus_nonexistant():
    assert EventBus.get("foo") == None
