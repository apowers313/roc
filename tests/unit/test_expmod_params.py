# mypy: disable-error-code="no-untyped-def"

"""Unit tests for ExpMod.params_dict()."""

import pytest

from roc.expmod import ExpMod, expmod_registry


@pytest.fixture(autouse=True)
def clean_expmod_state():
    orig_registry_snapshot = {k: set(v.keys()) for k, v in expmod_registry.items()}

    yield

    for modtype in list(expmod_registry.keys()):
        if modtype not in orig_registry_snapshot:
            del expmod_registry[modtype]
        else:
            for name in list(expmod_registry[modtype].keys()):
                if name not in orig_registry_snapshot[modtype]:
                    del expmod_registry[modtype][name]


class TestParamsDict:
    def test_params_dict_returns_public_attrs(self):
        class TestExpMod(ExpMod):
            modtype = "test-params-public"
            name = "test"
            threshold: float = 0.5

            def __init__(self) -> None:
                super().__init__()
                self.threshold = 0.5

        instance = expmod_registry["test-params-public"]["test"]
        result = instance.params_dict()
        assert "threshold" in result
        assert result["threshold"] == 0.5

    def test_params_dict_excludes_private(self):
        class TestExpMod(ExpMod):
            modtype = "test-params-private"
            name = "test"

            def __init__(self) -> None:
                super().__init__()
                self._internal = "secret"
                self.visible = 42

        instance = expmod_registry["test-params-private"]["test"]
        result = instance.params_dict()
        assert "_internal" not in result
        assert "visible" in result
        assert result["visible"] == 42

    def test_params_dict_excludes_callables(self):
        class TestExpMod(ExpMod):
            modtype = "test-params-callable"
            name = "test"

            def __init__(self) -> None:
                super().__init__()
                self.value = 10
                self.my_func = lambda: 42

        instance = expmod_registry["test-params-callable"]["test"]
        result = instance.params_dict()
        assert "value" in result
        assert "my_func" not in result

    def test_concrete_expmod_params(self):
        instance = expmod_registry["saliency-attenuation"]["linear-decline"]
        result = instance.params_dict()
        assert "capacity" in result
        assert "radius" in result
        assert "max_penalty" in result
        assert "max_attenuation" in result
