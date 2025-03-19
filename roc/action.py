"""The action module decides what action the agent should perform."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from roc.component import Component
from roc.event import Event, EventBus
from roc.expmod import ExpMod
from roc.graphdb import Node


@dataclass
class ActionRequest:
    """Communicates that the Gym is waiting for the agent to take an action."""


class TakeAction(Node):
    """Communicates back to the Gym which cation to take."""

    action: Any


ActionData = ActionRequest | TakeAction
ActionEvent = Event[ActionData]


class Action(Component):
    """Component for determining which action to take."""

    name: str = "action"
    type: str = "action"
    auto: bool = True

    bus = EventBus[ActionData]("action", cache_depth=10)

    def __init__(self) -> None:
        super().__init__()
        self.action_bus_conn = self.connect_bus(self.bus)
        self.action_bus_conn.listen(self.action_request)

    def event_filter(self, e: ActionEvent) -> bool:
        return isinstance(e.data, ActionRequest)

    def action_request(self, e: ActionEvent) -> None:
        action = DefaultActionExpMod.get(default="pass").get_action()
        actevt = TakeAction(action=action)
        self.action_bus_conn.send(actevt)


class DefaultActionExpMod(ExpMod):
    modtype = "action"

    @abstractmethod
    def get_action(self) -> int: ...


# @DefaultActionExpMod.register("pass")
class DefaultActionPass(DefaultActionExpMod):
    name = "pass"

    def get_action(self) -> int:
        """Default action for Nethack that passes (the `.` character in the game)"""
        return 19
