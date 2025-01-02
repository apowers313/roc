from __future__ import annotations

import importlib.util
import sys
from abc import abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Callable, Self, cast

from roc.config import Config

expmod_registry: dict[str, dict[str, ExpMod]] = defaultdict(dict)
expmod_modtype_current: dict[str, str | None] = defaultdict(lambda: None)
expmod_loaded: dict[str, ModuleType] = {}


class ExpMod:
    modtype: str

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "modtype"):
            raise NotImplementedError(f"{cls} must implement class attribute 'modtype'")

    @staticmethod
    def register(name: str) -> Callable[[type[ExpMod]], type[ExpMod]]:
        def register_decorator(cls: type[ExpMod]) -> type[ExpMod]:
            if name in expmod_registry[cls.modtype]:
                raise Exception(
                    f"ExpMod.register attempting to register duplicate name '{name}' for module '{cls.modtype}'"
                )
            expmod_registry[cls.modtype][name] = cls()

            return cls

        return register_decorator

    @classmethod
    def get(cls, default: str | None = None) -> Self:
        modtype = cls.modtype
        name: str | None = (
            expmod_modtype_current[modtype]
            if expmod_modtype_current[modtype] is not None
            else default
        )
        if name is None:
            raise Exception(f"ExpMod couldn't get module for type: '{modtype}'")

        return cast(Self, expmod_registry[modtype][name])

    @classmethod
    def set(cls, name: str, modtype: str | None = None) -> None:
        if modtype is None:
            modtype = cls.modtype

        if modtype not in expmod_registry:
            raise Exception(f"ExpMod.set can't find module for type: '{modtype}'")

        if name not in expmod_registry[modtype]:
            raise Exception(
                f"ExpMod.set can't find module for name: '{name}' in module '{modtype}'"
            )

        expmod_modtype_current[modtype] = name

    @staticmethod
    def import_file(filename: str, basepath: str = "") -> ModuleType:
        module_name = f"roc:expmod:{filename}"
        filepath = Path(basepath) / filename

        spec = importlib.util.spec_from_file_location(module_name, filepath)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module

    @staticmethod
    def init() -> None:
        settings = Config.get()

        mods = settings.expmods.copy()
        basepaths = settings.expmod_dirs.copy()
        basepaths.insert(0, "")

        # load module files
        missing_mods: list[str] = []
        for base in basepaths:
            for mod in mods:
                file = mod if mod.endswith(".py") else mod + ".py"
                try:
                    expmod_loaded[mod] = ExpMod.import_file(file, base)
                except FileNotFoundError:
                    missing_mods.append(mod)
            mods = missing_mods.copy()
            missing_mods.clear()

        if len(mods) > 0:
            raise FileNotFoundError(f"could not load experiment modules: {mods}")

        # set modules
        use_mods = [m.split(":") for m in settings.expmods_use]
        mod_name_count = Counter([m[0] for m in use_mods])
        duplicate_names = {k: v for k, v in mod_name_count.items() if v > 1}
        if len(duplicate_names) > 0:
            dupes = ", ".join(duplicate_names.keys())
            raise Exception(f"ExpMod.init found multiple attempts to set the same modules: {dupes}")

        for mod_tn in use_mods:
            t, n = mod_tn
            ExpMod.set(name=n, modtype=t)


class DefaultActionExpMod(ExpMod):
    modtype = "action"

    @abstractmethod
    def get_action(self) -> int: ...
