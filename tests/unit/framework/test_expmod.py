# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/framework/expmod.py."""

import pytest

from roc.framework.config import Config
from roc.framework.expmod import (
    ExpMod,
    ExpModConfig,
    ExpModDependencyCycleError,
    ExpModDependencyError,
    SharedConfigGroup,
    expmod_modtype_current,
    expmod_registry,
)


class TestExpModInitSubclass:
    def test_registers_in_registry(self):
        class TestRegisteredMod(ExpMod):
            modtype = "test-register-check"
            name = "test-entry"

        assert "test-register-check" in expmod_registry
        assert "test-entry" in expmod_registry["test-register-check"]

    def test_raises_on_missing_modtype(self):
        with pytest.raises(NotImplementedError, match="must implement class attribute 'modtype'"):

            class BadMod(ExpMod):
                name = "bad"

    def test_raises_on_duplicate(self):
        class OriginalMod(ExpMod):
            modtype = "test-dup-check"
            name = "original"

        _ = OriginalMod
        with pytest.raises(Exception, match="duplicate name"):

            class DuplicateMod(ExpMod):
                modtype = "test-dup-check"
                name = "original"


class TestExpModGet:
    def test_with_default(self):
        class TestGetMod(ExpMod):
            modtype = "test-get-default"
            name = "default-entry"

        _ = TestGetMod

        # Create a subclass reference to call .get() through the modtype base.
        class TestGetBase(ExpMod):
            modtype = "test-get-default"

        # Can't subclass directly — go through ExpMod.get via the fake class
        result = ExpMod.get.__func__(  # type: ignore[attr-defined]
            type("FakeExpMod", (), {"modtype": "test-get-default"}), default="default-entry"
        )
        assert result is expmod_registry["test-get-default"]["default-entry"]

    def test_missing_raises(self):
        with pytest.raises(Exception, match="couldn't get module"):

            class SomeExpMod(ExpMod):
                modtype = "test-get-missing"
                name = "placeholder-get-missing"

            expmod_modtype_current["test-get-missing"] = None
            SomeExpMod.get()


class TestExpModSet:
    def test_valid_set(self):
        class TestSetMod(ExpMod):
            modtype = "test-set-valid"
            name = "set-entry"

        _ = TestSetMod
        ExpMod.set(name="set-entry", modtype="test-set-valid")
        assert expmod_modtype_current["test-set-valid"] == "set-entry"

    def test_invalid_modtype_raises(self):
        with pytest.raises(Exception, match="can't find module for type"):
            ExpMod.set(name="foo", modtype="nonexistent-type-xyz")

    def test_invalid_name_raises(self):
        class TestInvalidNameMod(ExpMod):
            modtype = "test-invalid-name"
            name = "exists"

        _ = TestInvalidNameMod
        with pytest.raises(Exception, match="can't find module for name"):
            ExpMod.set(name="nonexistent-name-xyz", modtype="test-invalid-name")


class TestExpModInit:
    def test_activates_empty_use(self):
        settings = Config.get()
        settings.expmods_use = []
        ExpMod.init()

    def test_detects_duplicate_mod_use(self):
        settings = Config.get()
        settings.expmods_use = [("action", "pass"), ("action", "weighted")]

        with pytest.raises(ValueError, match="multiple attempts to set the same modules"):
            ExpMod.init()


class TestExpModConfigSchema:
    def test_config_defaults_populated_on_activation(self):
        class MyCfg(ExpModConfig):
            threshold: float = 0.25

        class MyBase(ExpMod):
            modtype = "test-cfg-defaults"

        class MyImpl(MyBase):
            name = "impl"
            config_schema = MyCfg

        _ = MyImpl

        settings = Config.get()
        settings.expmods_use = [("test-cfg-defaults", "impl")]
        settings.expmod_config = {}

        ExpMod.init()

        inst = expmod_registry["test-cfg-defaults"]["impl"]
        assert isinstance(inst.config, MyCfg)
        assert inst.config.threshold == pytest.approx(0.25)

    def test_config_overrides_from_settings(self):
        class MyCfg(ExpModConfig):
            threshold: float = 0.25

        class MyBase(ExpMod):
            modtype = "test-cfg-override"

        class MyImpl(MyBase):
            name = "impl"
            config_schema = MyCfg

        _ = MyImpl

        settings = Config.get()
        settings.expmods_use = [("test-cfg-override", "impl")]
        settings.expmod_config = {"test-cfg-override.impl": {"threshold": 0.9}}

        ExpMod.init()

        inst = expmod_registry["test-cfg-override"]["impl"]
        assert inst.config.threshold == pytest.approx(0.9)  # type: ignore[attr-defined]

    def test_invalid_override_raises(self):
        class MyCfg(ExpModConfig):
            threshold: float = 0.25

        class MyBase(ExpMod):
            modtype = "test-cfg-invalid"

        class MyImpl(MyBase):
            name = "impl"
            config_schema = MyCfg

        _ = MyImpl

        settings = Config.get()
        settings.expmods_use = [("test-cfg-invalid", "impl")]
        settings.expmod_config = {"test-cfg-invalid.impl": {"nope": 1}}  # field doesn't exist

        with pytest.raises(ValueError, match="invalid config"):
            ExpMod.init()


class TestSharedConfig:
    def test_shared_group_instance_reused(self):
        class Group(SharedConfigGroup):
            group_name = "test-group-A"
            k: int = 7

        class MyBase(ExpMod):
            modtype = "test-shared-A"

        class MyImplA(MyBase):
            name = "a"
            shared_config_schemas = (Group,)

        class MyBaseB(ExpMod):
            modtype = "test-shared-B"

        class MyImplB(MyBaseB):
            name = "b"
            shared_config_schemas = (Group,)

        _ = (MyImplA, MyImplB)

        settings = Config.get()
        settings.expmods_use = [("test-shared-A", "a"), ("test-shared-B", "b")]
        settings.expmod_config = {"shared.test-group-A": {"k": 42}}

        ExpMod.init()

        a = expmod_registry["test-shared-A"]["a"]
        b = expmod_registry["test-shared-B"]["b"]
        assert a.shared_configs["test-group-A"] is b.shared_configs["test-group-A"]
        assert a.shared_configs["test-group-A"].k == 42  # type: ignore[attr-defined]


class TestExpModDependencies:
    def test_satisfied_deps_ok(self):
        class DepBase(ExpMod):
            modtype = "test-dep-base"

        class DepImpl(DepBase):
            name = "dep-impl"

        class UserBase(ExpMod):
            modtype = "test-dep-user"

        class UserImpl(UserBase):
            name = "user-impl"
            depends_on = (("test-dep-base", "dep-impl"),)

        _ = (DepImpl, UserImpl)

        settings = Config.get()
        settings.expmods_use = [("test-dep-base", "dep-impl"), ("test-dep-user", "user-impl")]
        settings.expmod_config = {}

        ExpMod.init()

    def test_missing_dep_raises(self):
        class MissBase(ExpMod):
            modtype = "test-dep-miss-base"

        class MissImpl(MissBase):
            name = "miss-impl"

        class UserBase(ExpMod):
            modtype = "test-dep-miss-user"

        class UserImpl(UserBase):
            name = "user-impl"
            depends_on = (("test-dep-miss-base", "miss-impl"),)

        _ = (MissImpl, UserImpl)

        settings = Config.get()
        settings.expmods_use = [("test-dep-miss-user", "user-impl")]  # missing dep
        settings.expmod_config = {}

        with pytest.raises(ExpModDependencyError, match="not satisfied"):
            ExpMod.init()

    def test_unregistered_dep_raises(self):
        class UserBase(ExpMod):
            modtype = "test-dep-unreg-user"

        class UserImpl(UserBase):
            name = "user-impl"
            depends_on = (("does-not-exist", "also-no"),)

        _ = UserImpl

        settings = Config.get()
        settings.expmods_use = [("test-dep-unreg-user", "user-impl")]
        settings.expmod_config = {}

        with pytest.raises(ExpModDependencyError, match="not registered"):
            ExpMod.init()

    def test_transitive_deps_checked(self):
        class LeafBase(ExpMod):
            modtype = "test-dep-leaf"

        class LeafImpl(LeafBase):
            name = "leaf"

        class MidBase(ExpMod):
            modtype = "test-dep-mid"

        class MidImpl(MidBase):
            name = "mid"
            depends_on = (("test-dep-leaf", "leaf"),)

        class RootBase(ExpMod):
            modtype = "test-dep-root"

        class RootImpl(RootBase):
            name = "root"
            depends_on = (("test-dep-mid", "mid"),)

        _ = (LeafImpl, MidImpl, RootImpl)

        settings = Config.get()
        # Include mid but not leaf -> transitive chain broken
        settings.expmods_use = [("test-dep-root", "root"), ("test-dep-mid", "mid")]
        settings.expmod_config = {}

        with pytest.raises(ExpModDependencyError, match="not satisfied"):
            ExpMod.init()

    def test_cycle_detected(self):
        class ABase(ExpMod):
            modtype = "test-dep-cycle-A"

        class AImpl(ABase):
            name = "a"
            depends_on = (("test-dep-cycle-B", "b"),)

        class BBase(ExpMod):
            modtype = "test-dep-cycle-B"

        class BImpl(BBase):
            name = "b"
            depends_on = (("test-dep-cycle-A", "a"),)

        _ = (AImpl, BImpl)

        settings = Config.get()
        settings.expmods_use = [("test-dep-cycle-A", "a"), ("test-dep-cycle-B", "b")]
        settings.expmod_config = {}

        with pytest.raises(ExpModDependencyCycleError, match="cycle"):
            ExpMod.init()
