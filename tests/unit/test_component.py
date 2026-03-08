# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/component.py."""

from unittest.mock import MagicMock

import pytest

from roc.component import (
    Component,
    ComponentId,
    ComponentKey,
    ComponentName,
    ComponentType,
    _component_registry_key,
    component_registry,
    component_set,
    loaded_components,
)
from roc.event import EventBus, eventbus_names


@pytest.fixture(autouse=True)
def restore_component_state():
    """Save and restore component registry and loaded components around each test."""
    saved_registry = component_registry.copy()
    saved_loaded = loaded_components.copy()
    saved_bus_names = eventbus_names.copy()
    # Snapshot existing component_set weakrefs
    saved_set_refs = list(component_set)

    yield

    # Restore
    component_registry.clear()
    component_registry.update(saved_registry)
    loaded_components.clear()
    loaded_components.update(saved_loaded)
    eventbus_names.clear()
    eventbus_names.update(saved_bus_names)
    # Clean up any components we created
    for c in list(component_set):
        if c not in saved_set_refs:
            component_set.discard(c)


class TestComponentId:
    def test_str(self):
        cid = ComponentId("mytype", "myname")
        assert str(cid) == "myname:mytype"

    def test_equality(self):
        a = ComponentId("type1", "name1")
        b = ComponentId("type1", "name1")
        c = ComponentId("type2", "name1")
        assert a == b
        assert a != c

    def test_named_tuple_fields(self):
        cid = ComponentId("t", "n")
        assert cid.type == "t"
        assert cid.name == "n"


class TestComponentRegistryKey:
    def test_creates_key(self):
        key = _component_registry_key("myname", "mytype")
        assert key == ComponentKey((ComponentName("myname"), ComponentType("mytype")))


class TestComponentInitSubclass:
    def test_registers_in_registry(self):
        class _TestComp1(Component):
            name = "unit_test_comp_1"
            type = "unit_test"

        key = _component_registry_key("unit_test_comp_1", "unit_test")
        assert key in component_registry
        assert component_registry[key] is _TestComp1
        # cleanup
        Component.deregister("unit_test_comp_1", "unit_test")

    def test_raises_if_name_unset(self):
        with pytest.raises(Exception, match="name is unspecified"):

            class _BadNameComp(Component):
                type = "unit_test"

    def test_raises_if_type_unset(self):
        with pytest.raises(Exception, match="type is unspecified"):

            class _BadTypeComp(Component):
                name = "bad_type_comp"

    def test_raises_on_duplicate_registration(self):
        class _DupComp1(Component):
            name = "dup_comp"
            type = "unit_test_dup"

        with pytest.raises(ValueError, match="duplicate component name"):

            class _DupComp2(Component):
                name = "dup_comp"
                type = "unit_test_dup"

        Component.deregister("dup_comp", "unit_test_dup")


class TestComponentGetAndDeregister:
    def test_get_retrieves_and_instantiates(self):
        class _GetComp(Component):
            name = "get_comp"
            type = "unit_test_get"

            def __init__(self):
                super().__init__()
                self.created = True

        c = Component.get("get_comp", "unit_test_get")
        assert isinstance(c, _GetComp)
        assert c.created is True
        c.shutdown()
        component_set.discard(c)
        Component.deregister("get_comp", "unit_test_get")

    def test_deregister_removes(self):
        class _DeregComp(Component):
            name = "dereg_comp"
            type = "unit_test_dereg"

        key = _component_registry_key("dereg_comp", "unit_test_dereg")
        assert key in component_registry
        Component.deregister("dereg_comp", "unit_test_dereg")
        assert key not in component_registry


class TestComponentInstance:
    def test_connect_bus_raises_on_duplicate(self):
        class _BusComp(Component):
            name = "bus_comp"
            type = "unit_test_bus"

        comp = _BusComp()
        bus = EventBus[int]("test_dup_bus_conn")
        comp.connect_bus(bus)
        with pytest.raises(ValueError, match="duplicate connection"):
            comp.connect_bus(bus)

        comp.shutdown()
        component_set.discard(comp)
        Component.deregister("bus_comp", "unit_test_bus")
        eventbus_names.discard("test_dup_bus_conn")

    def test_event_filter_excludes_self(self):
        class _FilterComp(Component):
            name = "filter_comp"
            type = "unit_test_filter"

        comp = _FilterComp()
        mock_event_self = MagicMock()
        mock_event_self.src_id = comp.id
        mock_event_other = MagicMock()
        mock_event_other.src_id = ComponentId("other_type", "other_name")

        assert comp.event_filter(mock_event_self) is False
        assert comp.event_filter(mock_event_other) is True

        comp.shutdown()
        component_set.discard(comp)
        Component.deregister("filter_comp", "unit_test_filter")

    def test_id_property(self):
        class _IdComp(Component):
            name = "id_comp"
            type = "unit_test_id"

        comp = _IdComp()
        assert comp.id == ComponentId("unit_test_id", "id_comp")

        comp.shutdown()
        component_set.discard(comp)
        Component.deregister("id_comp", "unit_test_id")

    def test_duplicate_instance_raises(self):
        class _DupInstComp(Component):
            name = "dup_inst_comp"
            type = "unit_test_dup_inst"

        comp1 = _DupInstComp()
        with pytest.raises(Exception, match="component already exists"):
            _DupInstComp()

        comp1.shutdown()
        component_set.discard(comp1)
        Component.deregister("dup_inst_comp", "unit_test_dup_inst")


class TestComponentCountAndLoaded:
    def test_get_component_count(self):
        initial = Component.get_component_count()

        class _CountComp(Component):
            name = "count_comp"
            type = "unit_test_count"

        comp = _CountComp()
        assert Component.get_component_count() == initial + 1

        comp.shutdown()
        component_set.discard(comp)
        Component.deregister("count_comp", "unit_test_count")

    def test_get_loaded_components(self):
        # loaded_components is a separate dict for components loaded via Component.init()
        # Should return list of "name:type" strings
        result = Component.get_loaded_components()
        assert isinstance(result, list)


class TestComponentShutdownAndReset:
    def test_shutdown_closes_bus_connections(self):
        class _ShutComp(Component):
            name = "shut_comp"
            type = "unit_test_shut"

        comp = _ShutComp()
        bus = EventBus[int]("shut_bus")
        comp.connect_bus(bus)
        # Should not raise
        comp.shutdown()
        component_set.discard(comp)
        Component.deregister("shut_comp", "unit_test_shut")
        eventbus_names.discard("shut_bus")

    def test_reset_shuts_down_all(self):
        class _ResetComp(Component):
            name = "reset_comp"
            type = "unit_test_reset"

        comp = _ResetComp()
        loaded_components[
            (ComponentName("reset_comp"), ComponentType("unit_test_reset"))
        ] = comp

        Component.reset()
        assert len(loaded_components) == 0

        component_set.discard(comp)
        Component.deregister("reset_comp", "unit_test_reset")
