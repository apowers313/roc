"""The action module decides what action the agent should perform."""

from dataclasses import dataclass
from typing import Any

from .component import Component, register_component
from .event import Event, EventBus
from .expmod import DefaultActionExpMod


@dataclass
class ActionRequest:
    """Communicates that the Gym is waiting for the agent to take an action."""


@dataclass
class TakeAction:
    """Communicates back to the Gym which cation to take."""

    action: Any


ActionData = ActionRequest | TakeAction
ActionEvent = Event[ActionData]


@register_component("action", "action", auto=True)
class Action(Component):
    """Component for determining which action to take."""

    bus = EventBus[ActionData]("action", cache_depth=10)

    def __init__(self) -> None:
        super().__init__()
        self.action_bus_conn = self.connect_bus(self.bus)
        self.action_bus_conn.listen(self.action_request)

    def event_filter(self, e: ActionEvent) -> bool:
        return isinstance(e.data, ActionRequest)

    def action_request(self, e: ActionEvent) -> None:
        action = DefaultActionExpMod.get(default="pass").get_action()
        actevt = TakeAction(action)
        self.action_bus_conn.send(actevt)


@DefaultActionExpMod.register("pass")
class DefaultActionPass(DefaultActionExpMod):
    def get_action(self) -> int:
        """Default action for Nethack that passes (the `.` character in the game)"""
        return 19
