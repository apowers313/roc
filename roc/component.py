from abc import ABC, abstractmethod


class Component(ABC):
    """An abstract component class for building pieces of ROC that will talk to each other."""

    def __init__(self, name: str, type: str):
        """Component constructor.

        Args:
            name (str): Name of the component. Mostly used for eventing.
            type (str): Type of the component. Will be set by the concrete class. Mostly used for eventing.
        """
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        """Getter for the name of the component

        Returns:
            str: the name of the component

        Example:
            >>> c = Component("foo", "bar")
            >>> print(c.name)
            foo
        """
        return self._name

    @property
    def type(self) -> str:
        """Getter for the type of the component

        Returns:
            str: the type of the component

        Example:
            >>> c = Component("foo", "bar")
            >>> print(c.type)
            bar
        """
        return self._type
