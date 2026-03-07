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
