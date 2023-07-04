from abc import ABC, abstractmethod


class Component(ABC):
    def __init__(self, name: str, type: str):
        self._name = name
        self._type = type

    @property
    def name(self):
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
    def type(self):
        """Getter for the type of the component

        Returns:
            str: the type of the component

        Example:
            >>> c = Component("foo", "bar")
            >>> print(c.type)
            bar
        """
        return self._type
