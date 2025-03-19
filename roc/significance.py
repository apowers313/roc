from typing import Any

from .component import Component
from .config import Config
from .event import Event, EventBus
from .intrinsic import Intrinsic, IntrinsicData, IntrinsicEvent


class SignificanceData:
    def __init__(self, significance: float) -> None:
        self.significance = significance


SignificanceEvent = Event[SignificanceData]


class Significance(Component):
    name: str = "significance"
    type: str = "significance"
    auto: bool = True
    bus = EventBus[SignificanceData]("significance")

    def __init__(self) -> None:
        super().__init__()
        self.significance_conn = self.connect_bus(Significance.bus)
        self.intrinsic_conn = self.connect_bus(Intrinsic.bus)
        self.intrinsic_conn.listen(self.do_significance)

    def event_filter(self, e: Event[Any]) -> bool:
        return isinstance(e.data, IntrinsicData)

    def do_significance(self, e: IntrinsicEvent) -> None:
        significance = 0.0
        settings = Config.get()
        assert isinstance(e.data, IntrinsicData)

        normalized_intrinsics = e.data.normalized_intrinsics
        for name, val in normalized_intrinsics.items():
            weight = (
                settings.significance_weights[name]
                if name in settings.significance_weights
                else 1.0
            )
            significance += val * weight

        self.significance_conn.send(SignificanceData(significance))
