# mypy: disable-error-code="no-untyped-def"

from typing import Any, Generator

import pytest

from roc.config import Config
from roc.expmod import ExpMod, expmod_loaded, expmod_modtype_current, expmod_registry


class TestExpMod:
    @pytest.fixture(scope="module", autouse=True)
    def expmod_restore_registry(self) -> Generator[None, None, None]:
        from roc.expmod import expmod_loaded, expmod_registry

        orig_registry = expmod_registry
        orig_loaded = expmod_loaded

        yield

        expmod_registry = orig_registry
        expmod_loaded = orig_loaded

    @pytest.fixture(autouse=True)
    def expmod_clear_current(self) -> Generator[None, None, None]:
        expmod_registry.clear()
        expmod_modtype_current.clear()
        expmod_loaded.clear()

        yield

        expmod_registry.clear()
        expmod_modtype_current.clear()
        expmod_loaded.clear()

    @pytest.fixture
    def MyTestExpMod(self) -> Any:
        class MyTestExpMod(ExpMod):
            modtype = "test"
            name = "foo"

        return MyTestExpMod

    def test_exists(self, MyTestExpMod) -> None:
        MyTestExpMod()

    def test_set(self, MyTestExpMod) -> None:
        MyTestExpMod.set("foo")
        ret = MyTestExpMod.get()
        assert ret == expmod_registry["test"]["foo"]

    def test_get_with_default(self, MyTestExpMod) -> None:
        ret = MyTestExpMod.get("foo")
        assert ret == expmod_registry["test"]["foo"]

    def test_import_file(self) -> None:
        ret = ExpMod.import_file("testmod1.py", "tests/helpers")
        assert ret.foo == 42

    def test_import_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ExpMod.import_file("asdfa9ofwqhpfoihwe.py", "tests/helpers")

    def test_init(self):
        assert len(expmod_loaded) == 0
        settings = Config.get()
        settings.expmod_dirs = ["tests/helpers"]
        settings.expmods = ["testmod1"]
        settings.expmods_use = []

        ExpMod.init()

        assert len(expmod_loaded) == 1
        assert expmod_loaded["testmod1"].foo == 42

    def test_init_module_not_found(self):
        settings = Config.get()
        settings.expmod_dirs = ["tests/helpers"]
        settings.expmods = ["asdfascdasdf"]
        with pytest.raises(FileNotFoundError):
            ExpMod.init()

    def test_init_base_not_found(self):
        settings = Config.get()
        settings.expmod_dirs = ["asdfasdfwqefas"]
        settings.expmods = ["testmod1"]
        with pytest.raises(FileNotFoundError):
            ExpMod.init()

    def test_init_loads_module(self):
        assert len(expmod_loaded) == 0
        settings = Config.get()
        settings.expmod_dirs = ["tests/helpers"]
        settings.expmods = ["testmod2"]
        settings.expmods_use = ["test:testy"]

        ExpMod.init()

        assert len(expmod_loaded) == 1
        assert "testmod2" in expmod_loaded

        MyTestExpModClass = expmod_loaded["testmod2"].MyTestExpModClass
        ret = MyTestExpModClass.get("testy")
        assert ret.do_thing() == 31337

    def test_init_uses_module(self):
        assert len(expmod_loaded) == 0
        settings = Config.get()
        settings.expmod_dirs = ["tests/helpers"]
        settings.expmods = ["testmod2"]
        settings.expmods_use = ["test:testy"]
        assert len(expmod_modtype_current) == 0

        ExpMod.init()

        assert len(expmod_loaded) == 1
        assert len(expmod_modtype_current) == 1

        assert expmod_modtype_current["test"] == "testy"

    def test_init_uses_duplicate_module(self):
        assert len(expmod_loaded) == 0
        settings = Config.get()
        settings.expmod_dirs = ["tests/helpers"]
        settings.expmods = ["testmod2"]
        settings.expmods_use = ["test:testy", "test:testy"]
        assert len(expmod_modtype_current) == 0

        with pytest.raises(
            Exception, match="ExpMod.init found multiple attempts to set the same modules: test"
        ):
            ExpMod.init()

    def test_init_uses_missing_module(self):
        assert len(expmod_loaded) == 0
        settings = Config.get()
        settings.expmod_dirs = ["tests/helpers"]
        settings.expmods = ["testmod2"]
        settings.expmods_use = ["foo:bar"]
        assert len(expmod_modtype_current) == 0

        with pytest.raises(Exception, match="ExpMod.set can't find module for type: 'foo'"):
            ExpMod.init()

    def test_init_uses_missing_name(self):
        assert len(expmod_loaded) == 0
        settings = Config.get()
        settings.expmod_dirs = ["tests/helpers"]
        settings.expmods = ["testmod2"]
        settings.expmods_use = ["test:foo"]
        assert len(expmod_modtype_current) == 0

        with pytest.raises(
            Exception, match="ExpMod.set can't find module for name: 'foo' in module 'test'"
        ):
            ExpMod.init()
