"""Aggregates all Perception events and determines which locations / objects
should received focus
"""

from abc import ABC

import numpy as np
from pydantic import BaseModel
from skimage.feature import peak_local_max

from .component import Component, register_component
from .event import EventBus
from .graphdb import Node
from .location import DebugGrid, GenericGrid, Grid
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
    def __init__(self, grid: Grid) -> None:
        width = grid.width
        height = grid.height
        self.grid = grid
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
        max = 0
        for y in range(self.height):
            for x in range(self.width):
                curr = self.get_strength(x, y)
                if max < curr:
                    max = curr

        return max

    def get_strength(self, x: int, y: int) -> int:
        node_list = self.get_val(x, y)
        # TODO: not really sure that the strength should depend on the number of features
        ret = len(node_list)

        def add_strength(n: Node) -> None:
            nonlocal ret

            # TODO: this is pretty arbitrary and might be biased based on my
            # domain knowledge... I suspect I will come back and modify this
            # based on object recognition and other factors at some point in
            # the future
            if "Single" in n.labels:
                ret += 10
            if "Delta" in n.labels:
                ret += 15
            if "Motion" in n.labels:
                ret += 20

        for n in node_list:
            Node.walk(n, node_callback=add_strength, mode="dst")

        return ret

    def get_focus(self) -> None:
        max_str = self.get_max_strength()
        fkimg = np.array(
            [
                [self.get_strength(x, y) / max_str for y in range(self.height)]
                for x in range(self.width)
            ]
        )

        m = np.median(fkimg)
        # print("fkimg", fkimg[15:18, 4:7])
        coordinates = peak_local_max(fkimg, min_distance=5, threshold_rel=m)
        # print("coordinates", coordinates)

        for loc in coordinates:
            # print("loc", loc)
            x = loc[0]
            y = loc[1]
            p = self.grid.get_point(x, y)
            print("p", p)  # noqa: T201

        # return []

    def __str__(self) -> str:
        dg = DebugGrid(self.grid)
        max_str = self.get_max_strength()
        for p in self.grid.points():
            rel_strength = self.get_strength(p.x, p.y) / max_str
            color = DebugGrid.blue_to_red_hue(rel_strength)
            dg.set_style(p.x, p.y, back_brightness=1, back_hue=color)
        return str(dg)


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
            grid = Grid(e.data.chars)
            if not self.saliency_map:
                self.saliency_map = SaliencyMap(grid)
            else:
                self.saliency_map.grid = grid
            return

        # check to see if all feature extractors have settled
        if isinstance(e.data, Settled):
            self.settled.add(f"{e.src.name}:{e.src.type}")

            unsettled = set(FeatureExtractor.list()) - self.settled
            if len(unsettled) == 0:
                # self.att_conn.send(Settled())
                self.settled.clear()
                assert self.saliency_map
                self.saliency_map.get_focus()
                # print("done!")
                # self.saliency_map.clear()

            return

        # register each location in the saliency map
        assert isinstance(e.data, Node)
        f = e.data

        def try_add_loc(n: Node) -> None:
            if isinstance(n, ElementPoint):
                assert self.saliency_map
                self.saliency_map.add_val(n.x, n.y, n)

        # create saliency map
        Node.walk(f, node_callback=try_add_loc)

    # TODO: listen for vision events
    # TODO: select and emit a single event


# TODO: other attention classes


class CrossModalAttention(Attention):
    # TODO: listen for attention events
    # TODO: select and emit a single event
    pass
