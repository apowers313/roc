"""Runtime plugin system for swapping agent behaviors via configuration.

The ExpMod system lets you define interchangeable implementations of agent behaviors
(action selection, prediction strategies, etc.) and switch between them without modifying
core code. Implementations are selected by setting (modtype, name) pairs in config.

Architecture:
    - Define an abstract base by subclassing ExpMod with a ``modtype`` class attribute.
    - Define concrete implementations by subclassing the base with a ``name`` attribute.
    - Implementations auto-register on class definition via ``__init_subclass__``.
    - Consuming code calls ``MyBase.get(default="name").method()`` to dispatch to the
      active implementation.

Example:
    Define a new modtype and implementation::

        class MyExpMod(ExpMod):
            modtype = "my-feature"

            def do_thing(self) -> int: ...


        class MyImpl(MyExpMod):
            name = "simple"

            def do_thing(self) -> int:
                return 42

    Use it from consuming code::

        result = MyExpMod.get(default="simple").do_thing()

Configuration (in Config):
    - ``expmod_dirs``: directories to scan for module files (default: ``experiments/modules/``)
    - ``expmods``: filenames to dynamically import
    - ``expmods_use``: list of ``(modtype, name)`` tuples to activate

Per-ExpMod configuration:
    ExpMods that need tunable parameters should add individual fields to
    ``Config`` with a descriptive prefix (e.g. ``saliency_attenuation_radius``).
    The ExpMod reads these in its ``__init__`` via ``Config.get()``, falling
    back to class-attribute defaults if Config is not yet initialized.

    This keeps parameters discoverable, type-checked, and settable via
    environment variables (e.g. ``roc_saliency_attenuation_radius=5``).

    Example::

        # In config.py:
        my_expmod_threshold: float = 0.5


        # In the ExpMod:
        class MyImpl(MyExpMod):
            name = "fancy"
            threshold: float = 0.5  # default

            def __init__(self) -> None:
                super().__init__()
                try:
                    self.threshold = Config.get().my_expmod_threshold
                except Exception:
                    pass  # Config not initialized during import-time registration

Module-level state:
    - ``expmod_registry``: maps modtype -> name -> ExpMod instance
    - ``expmod_modtype_current``: tracks which name is active per modtype
    - ``expmod_loaded``: tracks dynamically imported module objects
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Self, cast

from roc.config import Config

expmod_registry: dict[str, dict[str, ExpMod]] = defaultdict(dict)
expmod_modtype_current: dict[str, str | None] = defaultdict(lambda: None)
expmod_loaded: dict[str, ModuleType] = {}


class ExpMod:
    """Base class for experiment modules that can be swapped at runtime.

    Subclasses register themselves automatically by ``modtype`` and ``name``, allowing
    different implementations to be selected via configuration. To create a new modtype,
    subclass ExpMod directly and set ``modtype``. To create an implementation, subclass
    the modtype base and set ``name``.

    Attributes:
        modtype: Identifies the category of behavior (e.g. "action", "prediction-candidate").
            Must be set by every subclass.
        name: Identifies a specific implementation within a modtype (e.g. "pass", "weighted").
            Must be set by concrete implementations (not required on abstract bases that
            directly subclass ExpMod).
    """

    modtype: str = str()
    name: str = str()

    def __init_subclass__(cls) -> None:
        """Auto-register subclasses into the expmod_registry.

        Validates that ``modtype`` is always set and that ``name`` is set for concrete
        implementations (classes that don't directly subclass ExpMod). Raises on duplicate
        registrations. Instantiates the class and stores it in ``expmod_registry[modtype][name]``.
        """
        if cls.modtype is ExpMod.modtype:
            raise NotImplementedError(f"{cls.__name__} must implement class attribute 'modtype'")

        if cls.name is ExpMod.name and ExpMod not in cls.__bases__:
            raise NotImplementedError(f"{cls.__name__} must implement class attribute 'name'")

        if cls.name in expmod_registry[cls.modtype]:
            raise ValueError(
                f"ExpMod.register attempting to register duplicate name '{cls.name}' for module '{cls.modtype}'"
            )
        expmod_registry[cls.modtype][cls.name] = cls()

    def params_dict(self) -> dict[str, Any]:
        """Return public, non-callable instance attributes as a dict.

        Returns:
            Dictionary of parameter names to their values.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and not callable(v)}

    @classmethod
    def get(cls, default: str | None = None) -> Self:
        """Returns the currently active module for this modtype.

        Args:
            default: Fallback module name if none has been set.

        Raises:
            Exception: If no module is set and no default is provided.

        Returns:
            The active ExpMod instance for this modtype.
        """
        modtype = cls.modtype
        name: str | None = (
            expmod_modtype_current[modtype]
            if expmod_modtype_current[modtype] is not None
            else default
        )
        if name is None:
            raise LookupError(f"ExpMod couldn't get module for type: '{modtype}'")

        return cast(Self, expmod_registry[modtype][name])

    @classmethod
    def set(cls, name: str, modtype: str | None = None) -> None:
        """Sets the active module for a given modtype.

        Args:
            name: The registered name of the module to activate.
            modtype: The module type category. Defaults to the calling class's modtype.

        Raises:
            Exception: If the modtype or name is not registered.
        """
        if modtype is None:
            modtype = cls.modtype

        if modtype not in expmod_registry:
            raise LookupError(f"ExpMod.set can't find module for type: '{modtype}'")

        if name not in expmod_registry[modtype]:
            raise LookupError(
                f"ExpMod.set can't find module for name: '{name}' in module '{modtype}'"
            )

        expmod_modtype_current[modtype] = name

    @staticmethod
    def import_file(filename: str, basepath: str = "") -> ModuleType:
        """Dynamically imports a Python file as an experiment module.

        Args:
            filename: The filename to import.
            basepath: Directory to prepend to the filename.

        Returns:
            The imported module.
        """
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
        """Load and activate experiment modules from configuration.

        Searches for module files listed in ``Config.expmods`` across all directories
        in ``Config.expmod_dirs`` (plus the current directory). Each directory is tried
        in order for any modules not yet found. After loading, activates implementations
        listed in ``Config.expmods_use``.

        Raises:
            FileNotFoundError: If any listed module file cannot be found in any search path.
            ValueError: If ``expmods_use`` contains duplicate modtype entries.
        """
        settings = Config.get()
        _load_expmod_files(settings)
        _activate_expmods(settings)


def _load_expmod_files(settings: Config) -> None:
    """Load module files from configured search paths."""
    mods = settings.expmods.copy()
    basepaths = settings.expmod_dirs.copy()
    basepaths.insert(0, "")

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


def _activate_expmods(settings: Config) -> None:
    """Activate experiment modules from configuration, checking for duplicates."""
    mod_name_count = Counter([m[0] for m in settings.expmods_use])
    duplicate_names = {k: v for k, v in mod_name_count.items() if v > 1}
    if len(duplicate_names) > 0:
        dupes = ", ".join(duplicate_names.keys())
        raise ValueError(f"ExpMod.init found multiple attempts to set the same modules: {dupes}")

    for mod_tn in settings.expmods_use:
        t, n = mod_tn
        ExpMod.set(name=n, modtype=t)
