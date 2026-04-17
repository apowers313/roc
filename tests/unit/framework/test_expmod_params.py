# mypy: disable-error-code="no-untyped-def"

"""Unit tests for ExpMod.params_dict()."""

import pytest

from roc.framework.expmod import ExpMod, ExpModConfig, SharedConfigGroup, expmod_registry


class TestParamsDict:
    def test_returns_schema_defaults(self):
        class Cfg(ExpModConfig):
            threshold: float = 0.5

        class Mod(ExpMod):
            modtype = "test-params-schema"

        class Impl(Mod):
            name = "test"
            config_schema = Cfg

        _ = Impl
        instance = expmod_registry["test-params-schema"]["test"]
        result = instance.params_dict()
        assert "threshold" in result
        assert result["threshold"] == pytest.approx(0.5)

    def test_excludes_internal_state(self):
        class Mod(ExpMod):
            modtype = "test-params-no-internal"

        class Impl(Mod):
            name = "test"

            def __init__(self) -> None:
                super().__init__()
                self._internal = "secret"

        _ = Impl
        instance = expmod_registry["test-params-no-internal"]["test"]
        result = instance.params_dict()
        assert "_internal" not in result

    def test_shared_group_included(self):
        class Group(SharedConfigGroup):
            group_name = "test-params-group"
            scale: int = 9

        class Mod(ExpMod):
            modtype = "test-params-with-shared"

        class Impl(Mod):
            name = "test"
            shared_config_schemas = (Group,)

            def __init__(self) -> None:
                super().__init__()
                # Simulate activation populating shared_configs
                self.shared_configs["test-params-group"] = Group()

        _ = Impl
        instance = expmod_registry["test-params-with-shared"]["test"]
        result = instance.params_dict()
        assert "shared.test-params-group" in result
        assert result["shared.test-params-group"] == {"scale": 9}

    def test_concrete_linear_decline_params(self):
        """``LinearDeclineAttenuation`` exposes its config schema via ``params_dict()``."""
        # Trigger registration
        from roc.expmods.saliency_attenuation import linear_decline  # noqa: F401

        instance = expmod_registry["saliency-attenuation"]["linear-decline"]
        result = instance.params_dict()
        assert "capacity" in result
        assert "radius" in result
        assert "max_penalty" in result
        assert "max_attenuation" in result
