"""The action module decides what action the agent should perform."""

from dataclasses import dataclass
from typing import Any, Callable

from .component import Component, register_component
from .event import Event, EventBus


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
        self.action_bus_conn.send(TakeAction(action=19))


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


# @register_default_action("random")
# def default_random() -> int:
#     """A default action for taking random Nethack actions."""
#     c = Action.get("action", "action")
#     if c.action_count is None:
#         raise ValueError("Trying to get action before actions have been configured")

#     return randrange(c.action_count)


# @register_default_action("weighted")
# def default_weighted() -> int:
#     """A default action for taking weighted NetHack actions, based on which
#     actions need to happen more often to ensure survival (moving, eating,
#     etc.)
#     """
#     from nle import FULL_ACTIONS

#     c = Action.get("action", "action")
#     if c.action_count is None:
#         raise ValueError("Trying to get action before actions have been configured")

#     weights: dict[int, int] = defaultdict(lambda: 1)
#     # XXX: this is FULL_ACTIONS, which has 86 members; the action list received
#     # only has 42
#     # use gym.env.action_space
#     #
#     # 0 N
#     # 1 E
#     # 2 S
#     # 3 W
#     # 4 NE
#     # 5 SE
#     # 6 SW
#     # 7 NW
#     # 8 N
#     # 9 E
#     # 10 S
#     # 11 W
#     # 12 NE
#     # 13 SE
#     # 14 SW
#     # 15 NW
#     # 16 UP
#     # 17 DOWN
#     # 18 WAIT
#     # 19 MORE
#     # 20 ADJUST
#     # 21 APPLY
#     # 22 ATTRIBUTES
#     # 23 CALL
#     # 24 CAST
#     # 25 CHAT
#     # 26 CLOSE
#     # 27 DIP
#     # 28 DROP
#     # 29 DROPTYPE
#     # 30 EAT
#     # 31 ENGRAVE
#     # 32 ENHANCE
#     # 33 ESC
#     # 34 FIGHT
#     # 35 FIRE
#     # 36 FORCE
#     # 37 INVENTORY
#     # 38 INVENTTYPE
#     # 39 INVOKE
#     # 40 JUMP
#     # 41 KICK
#     # 42 LOOK
#     # 43 LOOT
#     # 44 MONSTER
#     # 45 MOVE
#     # 46 MOVEFAR
#     # 47 OFFER
#     # 48 OPEN
#     # 49 PAY
#     # 50 PICKUP
#     # 51 PRAY
#     # 52 PUTON
#     # 53 QUAFF
#     # 54 QUIVER
#     # 55 READ
#     # 56 REMOVE
#     # 57 RIDE
#     # 58 RUB
#     # 59 RUSH
#     # 60 RUSH2
#     # 61 SEARCH
#     # 62 SEEARMOR
#     # 63 SEERINGS
#     # 64 SEETOOLS
#     # 65 SEETRAP
#     # 66 SEEWEAPON
#     # 67 SHELL
#     # 68 SIT
#     # 69 SWAP
#     # 70 TAKEOFF
#     # 71 TAKEOFFALL
#     # 72 THROW
#     # 73 TIP
#     # 74 TURN
#     # 75 TWOWEAPON
#     # 76 UNTRAP
#     # 77 VERSIONSHORT
#     # 78 WEAR
#     # 79 WIELD
#     # 80 WIPE
#     # 81 ZAP
#     # 82 PLUS
#     # 83 QUOTE
#     # 84 DOLLAR
#     # 85 SPACE
#     weights[FULL_ACTIONS.NE] = 25

#     return 19
