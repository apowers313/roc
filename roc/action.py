"""The action module decides what action the agent should perform."""

from random import randrange
from typing import Annotated, Callable, Literal

from pydantic import BaseModel, Field

from .component import Component, register_component
from .event import Event, EventBus


class ActionCount(BaseModel):
    """A Pydantic model for communicating the number of actions that can be
    taken.
    """

    type: Literal["action_count"] = "action_count"
    action_count: int


class ActionGo(BaseModel):
    """A Pydantic model for communicating that the Gym is waiting for the agent
    to take an action.
    """

    type: Literal["action_go"] = "action_go"
    go: bool


ActionData = Annotated[
    ActionCount | ActionGo,
    Field(discriminator="type"),
]

ActionEvent = Event[ActionData]


@register_component("action", "action", auto=True)
class Action(Component):
    """Component for determining which action to take."""

    bus = EventBus[ActionData]("action")

    def __init__(self) -> None:
        super().__init__()
        self.action_bus_conn = self.connect_bus(self.bus)

        # XXX: function because you can't type annotate an inline lambda
        def count_filter(e: ActionEvent) -> bool:
            return e.data.type == "action_count"

        def go_filter(e: ActionEvent) -> bool:
            return e.data.type == "action_go"

        self.action_bus_conn.listen(
            listener=self.recv_action_count,
            filter=count_filter,
        )
        self.action_count: None | int = None

    def recv_action_count(self, e: ActionEvent) -> None:
        if e.data.type != "action_count":
            raise Exception("bad data received in recv_action_count")

        self.action_count = e.data.action_count


ActionFn = Callable[[], int]
default_action_registry: dict[str, ActionFn] = {}


class register_default_action:
    """Decorator for registering potential default actions. Default actions are
    set in the configuration.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, fn: ActionFn) -> ActionFn:
        if self.name in default_action_registry:
            raise ValueError(f"Registering duplicate default action '{self.name}'")

        default_action_registry[self.name] = fn

        return fn


@register_default_action("pass")
def default_pass() -> int:
    """Default action for Nethack that passes (the `.` character in the game)"""
    return 19


@register_default_action("random")
def default_random() -> int:
    """A default action for taking random Nethack actions."""
    c = Action.get("action", "action")
    if c.action_count is None:
        raise ValueError("Trying to get action before actions have been configured")

    return randrange(c.action_count)
