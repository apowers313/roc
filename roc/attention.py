import math
from abc import ABC

from pydantic import BaseModel

from .component import Component, register_component
from .event import EventBus
from .graphdb import Node
from .location import GenericGrid
from .perception import (
    ElementPoint,
    Feature,
    FeatureExtractor,
    Perception,
    PerceptionEvent,
    Settled,
    VisionData,
)


class VisionAttentionData(BaseModel):
    foo: str


AttentionData = VisionAttentionData


class Attention(Component, ABC):
    bus = EventBus[AttentionData]("attention")
    pass


class SaliencyMap(GenericGrid[list[Node]]):
    def __init__(self, width: int, height: int) -> None:
        map: list[list[list[Node]]] = [[list() for col in range(width)] for row in range(height)]
        super().__init__(map)

    def clear(self) -> None:
        for val in self:
            val.clear()

    @property
    def size(self) -> int:
        sz = 0
        for val in self:
            sz = sz + len(val)

        return sz

    def add_val(self, x: int, y: int, val: Node) -> None:
        node_list = self.get_val(x, y)
        node_list.append(val)

    def get_max_strength(self) -> int:
        # if hasattr(self, "_cached_strength"):
        #     return self._cached_strength  # type: ignore

        max = 0
        for loc in self:
            sz = len(loc)
            if max < sz:
                max = sz

        # self._cached_strength = max
        return max

    def get_strength(self, x: int, y: int) -> int:
        # TODO: strength is current based on the number of nodes at a location
        # should be weighted for things like motion that have higher saliency
        return len(self.get_val(x, y))

    # def get_percent_strength(self, x: int, y: int) -> float:
    #     return self.get_strength(x, y) / self.get_max_strength()

    # def strengths(self) -> Iterator[Point]:
    #     """Iterate over all the points in the grid"""

    #     for y in range(self.height):
    #         for x in range(self.width):
    #             str = self.get_strength(x, y)
    #             yield Point(x, y, str)

    def __str__(self) -> str:
        EMPTY_CHR = 32  # ASCII space
        LIGHT_CHR = 9617  # Unicode 2591
        MED_CHR = 9618  # Unicode 2592
        DARK_CHR = 9619  # Unicode 2593
        FULL_CHR = 9608  # Unicode 2588
        shade_map: list[int] = [EMPTY_CHR, LIGHT_CHR, MED_CHR, DARK_CHR, FULL_CHR]

        ret = ""
        max_str = self.get_max_strength()
        for y in range(self.height):
            for x in range(self.width):
                rel_str = self.get_strength(x, y) / max_str
                shade_val = float_to_density(rel_str, shade_map)
                ret += chr(shade_val)
            ret += "\n"
        return ret


def float_to_density(v: float, density_map: list[int]) -> int:
    """Takes a float in the range 0 to 1 and returns the value in the density map."""

    max_idx = len(density_map)
    idx = math.floor(v * max_idx)
    if idx >= max_idx:
        idx = max_idx - 1
    return density_map[idx]


@register_component("vision", "attention")
class VisionAttention(Attention):
    saliency_map: SaliencyMap | None

    def __init__(self) -> None:
        super().__init__()
        self.pb_conn = self.connect_bus(Perception.bus)
        self.pb_conn.listen(self.do_attention)
        self.att_conn = self.connect_bus(Attention.bus)
        self.saliency_map = None
        self.settled: set[str] = set()

    # def get_all_feature_extractors(self) -> list[FeatureExtractor]:
    #     self.pb_conn.attached_bus

    def event_filter(self, e: PerceptionEvent) -> bool:
        # print("attention.event_filter", e)
        allow = (
            isinstance(e.data, Feature)
            or isinstance(e.data, Settled)
            or isinstance(e.data, VisionData)
        )
        # print("attention.event_filter passing:", allow)
        return allow

    def do_attention(self, e: PerceptionEvent) -> None:
        # create right-sized SaliencyMap based on VisionData
        if isinstance(e.data, VisionData):
            if not self.saliency_map:
                self.saliency_map = SaliencyMap(e.data.width, e.data.height)
            return

        # check to see if all feature extractors have settled
        if isinstance(e.data, Settled):
            self.settled.add(f"{e.src.name}:{e.src.type}")

            unsettled = set(FeatureExtractor.list()) - self.settled
            if len(unsettled) == 0:
                # self.att_conn.send(Settled())
                self.settled.clear()
                assert self.saliency_map
                # self.saliency_map.clear()

            return

        # register each location in the saliency map
        assert isinstance(e.data, Node)
        f = e.data

        def try_add_loc(n: Node) -> None:
            if isinstance(n, ElementPoint):
                assert self.saliency_map
                self.saliency_map.add_val(n.x, n.y, n)

        Node.walk(f, node_callback=try_add_loc)

    # TODO: listen for vision events
    # TODO: select and emit a single event


# TODO: other attention classes


class CrossModalAttention(Attention):
    # TODO: listen for attention events
    # TODO: select and emit a single event
    pass
