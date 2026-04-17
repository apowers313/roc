"""Runtime plugin system for swapping agent behaviors via configuration.

The ExpMod system lets you define interchangeable implementations of agent behaviors
(action selection, prediction strategies, etc.) and switch between them without modifying
core code. Implementations are selected by setting (modtype, name) pairs in config.

Architecture:
    - Define an abstract base by subclassing ``ExpMod`` with a ``modtype`` class attribute.
    - Define concrete implementations by subclassing the base with a ``name`` attribute.
    - Implementations auto-register on class definition via ``__init_subclass__``.
    - Consuming code calls ``MyBase.get(default="name").method()`` to dispatch to the
      active implementation.

Example:
    Define a new modtype and implementation::

        class MyExpMod(ExpMod):
            modtype = "my-feature"

            def do_thing(self) -> int: ...


        class MyConfig(ExpModConfig):
            threshold: float = 0.5


        class MyImpl(MyExpMod):
            name = "simple"
            config_schema = MyConfig

            def do_thing(self) -> int:
                return int(self.config.threshold * 100)

    Use it from consuming code::

        result = MyExpMod.get(default="simple").do_thing()

Configuration:
    Each ExpMod can declare a ``config_schema`` (a ``ExpModConfig`` subclass) with its
    private fields, and ``shared_config_schemas`` (a tuple of ``SharedConfigGroup``
    subclasses) for fields that multiple ExpMods read. Values come from the main
    ``Config.expmod_config`` dict:

        - Private: ``Config.expmod_config["<modtype>.<name>"]`` (dict of overrides)
        - Shared: ``Config.expmod_config["shared.<group_name>"]`` (dict of overrides)

    Shared groups are instantiated once per unique ``group_name`` and shared by every
    ExpMod that references that group. Defaults come from the Pydantic model.

Dependencies:
    An ExpMod may declare ``depends_on`` as a tuple of ``(modtype, name)`` pairs. At
    ``ExpMod.init()`` time every active ExpMod's dependency tree is walked transitively;
    any unsatisfied dependency or cycle raises.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, ClassVar, Iterator, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from roc.framework.config import Config


class ExpModConfig(BaseModel):
    """Base class for per-ExpMod configuration schemas."""

    model_config = ConfigDict(extra="forbid")


class SharedConfigGroup(ExpModConfig):
    """Configuration shared across multiple ExpMods.

    Subclasses must set ``group_name`` as a ``ClassVar[str]``. The same group_name
    may be referenced by multiple ExpMods; they all receive the same instance.
    """

    group_name: ClassVar[str] = ""


class ExpModDependencyError(Exception):
    """Raised when an ExpMod's declared dependencies are not satisfied by ``expmods_use``."""


class ExpModDependencyCycleError(ExpModDependencyError):
    """Raised when ``depends_on`` relationships among active ExpMods form a cycle."""


expmod_registry: dict[str, dict[str, "ExpMod"]] = defaultdict(dict)
expmod_modtype_current: dict[str, str | None] = defaultdict(lambda: None)
_shared_instances: dict[str, SharedConfigGroup] = {}


class ExpMod:
    """Base class for experiment modules that can be swapped at runtime.

    Subclasses register themselves automatically by ``modtype`` and ``name``. To create
    a new modtype, subclass ExpMod directly and set ``modtype``. To create an
    implementation, subclass the modtype base and set ``name``.

    Attributes:
        modtype: Identifies the category of behavior (e.g. "action").
        name: Identifies a specific implementation within a modtype (e.g. "pass").
        config_schema: Optional ``ExpModConfig`` subclass describing this ExpMod's
            private configuration. Defaults and validation come from the model.
        shared_config_schemas: Tuple of ``SharedConfigGroup`` subclasses this ExpMod
            reads configuration from.
        depends_on: Tuple of ``(modtype, name)`` pairs that must also be active.
    """

    modtype: str = str()
    name: str = str()

    config_schema: ClassVar[type[ExpModConfig] | None] = None
    shared_config_schemas: ClassVar[tuple[type[SharedConfigGroup], ...]] = ()
    depends_on: ClassVar[tuple[tuple[str, str], ...]] = ()

    def __init_subclass__(cls) -> None:
        if cls.modtype is ExpMod.modtype:
            raise NotImplementedError(f"{cls.__name__} must implement class attribute 'modtype'")

        if cls.name is ExpMod.name and ExpMod not in cls.__bases__:
            raise NotImplementedError(f"{cls.__name__} must implement class attribute 'name'")

        if cls.name in expmod_registry[cls.modtype]:
            raise ValueError(
                f"ExpMod.register attempting to register duplicate name '{cls.name}' "
                f"for module '{cls.modtype}'"
            )
        expmod_registry[cls.modtype][cls.name] = cls()

    def __init__(self) -> None:
        self.config: ExpModConfig = (
            self.config_schema() if self.config_schema is not None else ExpModConfig()
        )
        self.shared_configs: dict[str, SharedConfigGroup] = {}

    def params_dict(self) -> dict[str, Any]:
        """Return the resolved configuration as a flat dict (for logging/display).

        Private config fields appear under their own names; shared groups appear under
        ``shared.<group_name>`` mapped to the group's ``model_dump()``.
        """
        out: dict[str, Any] = {}
        if self.config_schema is not None:
            out.update(self.config.model_dump())
        for schema in self.shared_config_schemas:
            gname = schema.group_name
            group = self.shared_configs.get(gname)
            if group is not None:
                out[f"shared.{gname}"] = group.model_dump()
        return out

    @classmethod
    def get(cls, default: str | None = None) -> Self:
        """Return the active implementation for this modtype.

        Args:
            default: Fallback name if none has been activated via config.

        Raises:
            LookupError: If no module is active and no default is provided.
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
        """Activate an implementation for a given modtype.

        Args:
            name: The registered name of the module to activate.
            modtype: The module type category; defaults to the calling class's modtype.
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
    def init() -> None:
        """Activate experiment modules, load their configs, verify dependencies, and print.

        Idempotent: re-running activates the same set cleanly because ``ExpMod.set()``
        simply re-points the current name, and configs are rebuilt from main Config.

        Raises:
            ValueError: duplicate modtype entries in ``expmods_use`` or invalid config.
            ExpModDependencyError: unsatisfied dependency or dependency cycle among
                active ExpMods.
        """
        settings = Config.get()
        _activate_expmods(settings)
        _load_configs(settings)
        _check_dependencies()
        print_active_expmods()


def _activate_expmods(settings: Config) -> None:
    """Call ``ExpMod.set`` for every ``(modtype, name)`` entry in ``expmods_use``."""
    mod_name_count = Counter(m[0] for m in settings.expmods_use)
    duplicate_names = [k for k, v in mod_name_count.items() if v > 1]
    if duplicate_names:
        dupes = ", ".join(duplicate_names)
        raise ValueError(f"ExpMod.init found multiple attempts to set the same modules: {dupes}")

    for modtype, name in settings.expmods_use:
        ExpMod.set(name=name, modtype=modtype)


def _iter_active() -> Iterator[tuple[str, str]]:
    """Yield ``(modtype, name)`` for every currently activated ExpMod."""
    for modtype, name in expmod_modtype_current.items():
        if name is not None:
            yield (modtype, name)


def _load_configs(settings: Config) -> None:
    """Materialize per-ExpMod and shared config instances from ``settings.expmod_config``.

    Shared groups are instantiated once per unique ``group_name`` and shared across
    every active ExpMod that references that group.
    """
    _shared_instances.clear()

    for modtype, name in _iter_active():
        expmod = expmod_registry[modtype][name]

        if expmod.config_schema is not None:
            key = f"{modtype}.{name}"
            overrides = settings.expmod_config.get(key, {})
            try:
                expmod.config = expmod.config_schema(**overrides)
            except ValidationError as e:
                raise ValueError(
                    f"invalid config for ExpMod '{modtype}.{name}' "
                    f"(from Config.expmod_config[{key!r}]): {e}"
                ) from e

        expmod.shared_configs = {}
        for schema in expmod.shared_config_schemas:
            gname = schema.group_name
            if not gname:
                raise ValueError(
                    f"SharedConfigGroup {schema.__name__} used by ExpMod "
                    f"'{modtype}.{name}' is missing a non-empty 'group_name' ClassVar"
                )
            if gname not in _shared_instances:
                key = f"shared.{gname}"
                overrides = settings.expmod_config.get(key, {})
                try:
                    _shared_instances[gname] = schema(**overrides)
                except ValidationError as e:
                    raise ValueError(
                        f"invalid shared config '{gname}' (from Config.expmod_config[{key!r}]): {e}"
                    ) from e
            expmod.shared_configs[gname] = _shared_instances[gname]


def _check_dependencies() -> None:
    """Walk ``depends_on`` transitively for every active ExpMod and verify satisfaction."""
    active = set(_iter_active())
    for modtype, name in active:
        expmod = expmod_registry[modtype][name]
        _walk_deps(expmod, active, set(), [(modtype, name)])


def _walk_deps(
    expmod: ExpMod,
    active: set[tuple[str, str]],
    visited: set[tuple[str, str]],
    chain: list[tuple[str, str]],
) -> None:
    """Recursive dependency walk. Raises on unmet deps or cycles."""
    key = (expmod.modtype, expmod.name)
    if key in visited:
        cycle = " -> ".join(f"({t},{n})" for t, n in chain)
        raise ExpModDependencyCycleError(f"ExpMod dependency cycle: {cycle}")
    visited.add(key)

    for dep_modtype, dep_name in expmod.depends_on:
        if dep_modtype not in expmod_registry or dep_name not in expmod_registry[dep_modtype]:
            raise ExpModDependencyError(
                f"ExpMod '({expmod.modtype},{expmod.name})' declares dependency on "
                f"'({dep_modtype},{dep_name})' which is not registered"
            )
        if (dep_modtype, dep_name) not in active:
            chain_str = " -> ".join(f"({t},{n})" for t, n in chain)
            raise ExpModDependencyError(
                f"ExpMod dependency not satisfied: {chain_str} requires "
                f"({dep_modtype},{dep_name}) to be in Config.expmods_use"
            )
        dep = expmod_registry[dep_modtype][dep_name]
        _walk_deps(dep, active, visited, chain + [(dep_modtype, dep_name)])


def print_active_expmods() -> None:
    """Print a banner listing active ExpMods and their resolved configuration.

    Called at the end of ``ExpMod.init()`` to aid reproducibility: the exact
    (modtype, name, config) tuple for every active experiment is printed to stdout.
    """
    active = sorted(_iter_active())
    if not active:
        print("Active ExpMods: (none)")  # noqa: T201
        return

    print("Active ExpMods:")  # noqa: T201
    for modtype, name in active:
        expmod = expmod_registry[modtype][name]
        print(f"  {modtype} = {name}")  # noqa: T201
        if expmod.config_schema is not None:
            for field, value in expmod.config.model_dump().items():
                print(f"    {field} = {value}")  # noqa: T201
        for schema in expmod.shared_config_schemas:
            gname = schema.group_name
            group = expmod.shared_configs.get(gname)
            if group is None:
                continue
            print(f"    [shared:{gname}]")  # noqa: T201
            for field, value in group.model_dump().items():
                print(f"      {field} = {value}")  # noqa: T201
