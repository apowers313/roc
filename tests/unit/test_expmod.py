# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/expmod.py."""

import tempfile
from pathlib import Path

import pytest

from roc.expmod import ExpMod, expmod_loaded, expmod_modtype_current, expmod_registry


@pytest.fixture(autouse=True)
def clean_expmod_state():
    """Save and restore expmod global state between tests.

    Only removes entries added during the test -- does not blow away
    pre-existing entries that other modules may have registered.
    """
    orig_registry_snapshot = {k: set(v.keys()) for k, v in expmod_registry.items()}
    orig_current = dict(expmod_modtype_current)
    orig_loaded = dict(expmod_loaded)

    yield

    # Remove any entries added during the test (but don't remove pre-existing ones)
    for modtype in list(expmod_registry.keys()):
        if modtype not in orig_registry_snapshot:
            del expmod_registry[modtype]
        else:
            for name in list(expmod_registry[modtype].keys()):
                if name not in orig_registry_snapshot[modtype]:
                    del expmod_registry[modtype][name]

    # Restore current and loaded
    expmod_modtype_current.clear()
    expmod_modtype_current.update(orig_current)
    expmod_loaded.clear()
    expmod_loaded.update(orig_loaded)


class TestExpModInitSubclass:
    def test_registers_in_registry(self):
        # Create a new subclass and verify it was registered
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
        # Create a fresh modtype, then try to duplicate it
        class OriginalMod(ExpMod):
            modtype = "test-dup-check"
            name = "original"

        with pytest.raises(Exception, match="duplicate name"):

            class DuplicateMod(ExpMod):
                modtype = "test-dup-check"
                name = "original"


class TestExpModGet:
    def test_with_default(self):
        # Create a fresh modtype to test with
        class TestGetMod(ExpMod):
            modtype = "test-get-default"
            name = "default-entry"

        result = ExpMod.get.__func__(type("FakeExpMod", (), {"modtype": "test-get-default"}), default="default-entry")  # type: ignore[attr-defined]
        assert result is expmod_registry["test-get-default"]["default-entry"]

    def test_missing_raises(self):
        with pytest.raises(Exception, match="couldn't get module"):

            class SomeExpMod(ExpMod):
                modtype = "test-get-missing"
                name = "placeholder-get-missing"

            # Clear the current setting so no default is used
            expmod_modtype_current["test-get-missing"] = None
            SomeExpMod.get()


class TestExpModSet:
    def test_valid_set(self):
        # Create a fresh modtype to test with
        class TestSetMod(ExpMod):
            modtype = "test-set-valid"
            name = "set-entry"

        ExpMod.set(name="set-entry", modtype="test-set-valid")
        assert expmod_modtype_current["test-set-valid"] == "set-entry"

    def test_invalid_modtype_raises(self):
        with pytest.raises(Exception, match="can't find module for type"):
            ExpMod.set(name="foo", modtype="nonexistent-type-xyz")

    def test_invalid_name_raises(self):
        # Create a fresh modtype so the modtype exists but the name doesn't
        class TestInvalidNameMod(ExpMod):
            modtype = "test-invalid-name"
            name = "exists"

        with pytest.raises(Exception, match="can't find module for name"):
            ExpMod.set(name="nonexistent-name-xyz", modtype="test-invalid-name")


class TestExpModImportFile:
    def test_valid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("X = 42\n")
            f.flush()
            filepath = Path(f.name)

        mod = ExpMod.import_file(filepath.name, basepath=str(filepath.parent))
        assert mod.X == 42

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ExpMod.import_file("does_not_exist_12345.py")


class TestExpModInit:
    def test_loads_modules_from_config(self):
        from roc.config import Config

        settings = Config.get()
        settings.expmods = []
        settings.expmods_use = []

        # Should not raise
        ExpMod.init()

    def test_handles_missing_files(self):
        from roc.config import Config

        settings = Config.get()
        settings.expmods = ["nonexistent_module_xyz"]
        settings.expmod_dirs = []

        with pytest.raises(FileNotFoundError, match="could not load experiment modules"):
            ExpMod.init()

    def test_detects_duplicate_mod_use(self):
        from roc.config import Config

        settings = Config.get()
        settings.expmods = []
        settings.expmods_use = [("action", "pass"), ("action", "other")]

        with pytest.raises(Exception, match="multiple attempts to set the same modules"):
            ExpMod.init()
