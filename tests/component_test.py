# mypy: disable-error-code="no-untyped-def"


import pytest

from roc.component import (
    Component,
    component_registry,
    default_components,
    loaded_components,
)


class TestComponent:
    def test_component_count(self, empty_components):
        assert Component.get_component_count() == 0

    @pytest.mark.parametrize("requires_module", [["roc.feature_extractors.delta"]], indirect=True)
    def test_init(self, empty_components, requires_module):
        assert Component.get_component_count() == 0
        assert len(component_registry) > 0
        assert len(default_components) > 0
        Component.init()

        assert len(loaded_components) >= len(default_components)
        assert Component.get_component_count() == len(loaded_components)

    @pytest.mark.parametrize("requires_module", [["roc.feature_extractors.delta"]], indirect=True)
    def test_shutdown(self, empty_components, requires_module):
        assert Component.get_component_count() == 0
        Component.init()
        assert len(loaded_components) >= len(default_components)
        assert Component.get_component_count() == len(loaded_components)

    def test_connect_bus(self, fake_component, fake_bus):
        fake_component.connect_bus(fake_bus)

        assert len(fake_component.bus_conns) == 1


class TestComponentRegisterDecorator:
    def test_decorator(self, registered_test_component):
        n, t = registered_test_component
        reg_str = f"{n}:{t}"

        assert reg_str in component_registry

    def test_decorator_doc(self, registered_test_component):
        n, t = registered_test_component
        reg_str = f"{n}:{t}"

        assert reg_str in component_registry
        assert component_registry[reg_str].__doc__ == "This is a Bar doc"

    def test_decorator_creates_class(self, registered_test_component):
        n, t = registered_test_component
        reg_str = f"{n}:{t}"

        assert reg_str in component_registry
        c = component_registry[reg_str]()
        assert isinstance(c, Component)
        # assert isinstance(c, Bar)
        assert c.name == n
        assert c.type == t
