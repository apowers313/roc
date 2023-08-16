from typing import Annotated, Literal

from pydantic import BaseModel, Field
from reactivex import operators as op

from .component import Component
from .event import Event, EventBus


class ActionEmpty(BaseModel):
    type: Literal["action_empty"] = "action_empty"


class ActionCount(BaseModel):
    type: Literal["action_count"] = "action_count"
    action_count: int


ActionData = Annotated[
    ActionEmpty | ActionCount,
    Field(discriminator="type"),
]

action_bus = EventBus[ActionData]("action")
ActionEvent = Event[ActionData]


class ActionComponent(Component):
    def __init__(self) -> None:
        super().__init__("action", "action")
        self.action_bus_conn = action_bus.connect(self)

        def count_filter(e: ActionEvent) -> bool:
            return e.data.type == "action_count"

        self.action_bus_conn.subject.pipe(
            op.filter(count_filter),
        ).subscribe(self.recv_action_count)
        self.action_count: None | int = None

    def recv_action_count(self, e: ActionEvent) -> None:
        if e.data.type != "action_count":
            raise Exception("bad data received in recv_action_count")

        self.action_count = e.data.action_count
