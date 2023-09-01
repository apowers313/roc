from __future__ import annotations

from typing import Any, TypeVar, cast

from typing_extensions import Self

# from abc import ABC


class Component:
    """An abstract component class for building pieces of ROC that will talk to each other."""

    name: str = "<name unassigned>"
    type: str = "<type unassigned>"

    # def __init__(self, name: str, type: str):
    # """Component constructor.

    # Args:
    #     name (str): Name of the component. Mostly used for eventing.
    #     type (str): Type of the component. Will be set by the concrete class. Mostly used for \
    #         eventing.
    # """
    # # self._name = name
    # # self._type = type

    # @property
    # def name(self) -> str:
    #     """Getter for the name of the component

    #     Returns:
    #         str: the name of the component

    #     Example:
    #         >>> c = Component("foo", "bar")
    #         >>> print(c.name)
    #         foo
    #     """
    #     return self._name

    # @property
    # def type(self) -> str:
    #     """Getter for the type of the component

    #     Returns:
    #         str: the type of the component

    #     Example:
    #         >>> c = Component("foo", "bar")
    #         >>> print(c.type)
    #         bar
    #     """
    #     return self._type
    @classmethod
    def get(cls, name: str, type: str, *args: Any, **kwargs: Any) -> Self:
        reg_str = component_registry_key(name, type)
        return cast(Self, component_registry[reg_str](*args, **kwargs))

    @classmethod
    def clear_registry(cls) -> None:
        component_registry.clear()


WrappedComponentBase = TypeVar("WrappedComponentBase", bound=Component)
component_registry: dict[str, type[Component]] = {}


def component_registry_key(name: str, type: str) -> str:
    return f"{name}:{type}"


class register_component:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type

    def __call__(self, cls: type[Component]) -> type[Component]:
        global register_component
        reg_str = component_registry_key(self.name, self.type)
        if reg_str in component_registry:
            raise ValueError(f"Registering duplicate component name: '{self.name}'")
        component_registry[reg_str] = cls
        cls.name = self.name
        cls.type = self.type

        return cls

        # @functools.wraps(cls)
        # class WrappedComponent(cls):  # type: ignore [valid-type,misc]
        #     def __init__(self, *args: Any, **kwargs_orig: Any) -> None:
        #         kwargs = kwargs_orig.copy()
        #         kwargs["name"] = self.name
        #         super().__init__(*args, **kwargs)

        # return WrappedComponent
