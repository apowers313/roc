# mypy: disable-error-code="no-untyped-def"
from typing import Any

import pytest

from roc.component import (
    Component,
    component_registry,
    default_components,
    loaded_components,
)


def args(arg: int) -> Any:
    return pytest.mark.parametrize("testing_args", [arg], indirect=True)


class TestComponent:
    def test_component_count(self, empty_components):
        assert Component.get_component_count() == 0

    def test_init(self, empty_components):
        assert Component.get_component_count() == 0
        assert len(component_registry) > 0
        assert len(default_components) > 0
        Component.init()

        assert len(loaded_components) >= len(default_components)
        assert Component.get_component_count() == len(loaded_components)

    def test_shutdown(self, empty_components):
        assert Component.get_component_count() == 0
        Component.init()
        assert len(loaded_components) >= len(default_components)
        assert Component.get_component_count() == len(loaded_components)

    def test_connect_bus(self, fake_component, fake_bus):
        fake_component.connect_bus(fake_bus)

        assert len(fake_component.bus_conns) == 1

    # @pytest.mark.parametrize("testing_args", [42, 69, 7], indirect=True)
    @args(42)
    def test_args(self, testing_args):
        print("testing args is", testing_args)


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
