"""Aggregates all Perception events and determines which locations / objects
should received focus
"""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from skimage.feature import peak_local_max

from .component import Component, register_component
from .config import Config
from .event import EventBus
from .graphdb import Node
from .location import DebugGrid, IntGrid, NewGrid, Point
from .logger import logger
from .perception import (
    ElementPoint,
    Feature,
    FeatureExtractor,
    Perception,
    PerceptionEvent,
    Settled,
    VisionData,
)


@dataclass
class VisionAttentionData:
    focus_points: set[tuple[int, int]]
    saliency_map: SaliencyMap


AttentionData = VisionAttentionData


class Attention(Component, ABC):
    bus = EventBus[AttentionData]("attention")
    pass


class SaliencyMap(NewGrid[list[Node]]):
    grid: IntGrid | None

    def __new__(cls, grid: IntGrid | None = None) -> Self:
        settings = Config.get()
        my_shape = grid.shape if grid is not None else settings.observation_shape
        assert my_shape is not None
        obj = np.ndarray(my_shape, dtype=object).view(cls)
        for row, col in np.ndindex(my_shape):
            obj[row, col] = list()
        obj.grid = grid

        return obj

    def __array_finalize__(self, obj: npt.NDArray[Any] | None) -> None:
        if obj is None:
            return
        self.grid = getattr(obj, "grid", None)

    def __deepcopy__(self, memodict: object | None = None) -> SaliencyMap:
        sm = SaliencyMap(deepcopy(self.grid))
        for row, col in np.ndindex(self.shape):
            sm[row, col] = self[row, col].copy()
        return sm

    def clear(self) -> None:
        """Clears out all values from the SaliencyMap."""
        for row, col in np.ndindex(self.shape):
            self[row, col].clear()

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

    def feature_report(self) -> dict[str, int]:
        ret: dict[str, int] = dict()
        for row, col in np.ndindex(self.shape):
            node_list = self[row, col]
            for n in node_list:
                f = Feature.find_parent_feature(n)
                feature_name = type(f).__name__
                ret[feature_name] = 0 if feature_name not in ret else ret[feature_name] + 1

        return ret

    def get_focus(self) -> set[tuple[int, int]]:
        max_str = self.get_max_strength()

        # prevent divide by zero
        if max_str == 0:
            max_str = 1

        fkimg = np.array(
            [
                [self.get_strength(x, y) / max_str for y in range(self.height)]
                for x in range(self.width)
            ]
        )

        m = np.median(fkimg)
        coordinates = peak_local_max(fkimg, min_distance=5, threshold_rel=m)

        ret: set[tuple[int, int]] = {(loc[0], loc[1]) for loc in coordinates}

        return ret

    def __str__(self) -> str:
        assert self.grid is not None
        dg = DebugGrid(self.grid)
        max_str = self.get_max_strength()

        # prevent divide by zero
        if max_str == 0:
            max_str = 1

        for p in self.grid.points():
            rel_strength = self.get_strength(p.x, p.y) / max_str
            color = DebugGrid.blue_to_red_hue(rel_strength)
            dg.set_style(p.x, p.y, back_brightness=1, back_hue=color)
        return str(dg)


@register_component("vision", "attention", auto=True)
class VisionAttention(Attention):
    saliency_map: SaliencyMap

    def __init__(self) -> None:
        super().__init__()
        self.pb_conn = self.connect_bus(Perception.bus)
        self.pb_conn.listen(self.do_attention)
        self.att_conn = self.connect_bus(Attention.bus)
        self.saliency_map = SaliencyMap()
        self.settled: set[str] = set()

    def event_filter(self, e: PerceptionEvent) -> bool:
        allow = (
            isinstance(e.data, Feature)
            or isinstance(e.data, Settled)
            or isinstance(e.data, VisionData)
        )
        return allow

    def do_attention(self, e: PerceptionEvent) -> None:
        # create right-sized SaliencyMap based on VisionData
        if isinstance(e.data, VisionData):
            self.saliency_map.grid = IntGrid(e.data.chars)
            return

        # check to see if all feature extractors have settled
        if isinstance(e.data, Settled):
            self.settled.add(f"{e.src.name}:{e.src.type}")

            unsettled = set(FeatureExtractor.list()) - self.settled
            if len(unsettled) == 0:
                assert self.saliency_map is not None
                focus = self.saliency_map.get_focus()

                self.att_conn.send(
                    VisionAttentionData(
                        focus_points=self.saliency_map.get_focus(),
                        saliency_map=self.saliency_map,
                    )
                )

                # save state so that it can be inspected in Jupyter
                from .jupyter.state import states

                states.salency.set(deepcopy(self.saliency_map))

                # reset
                self.settled.clear()
                self.saliency_map = SaliencyMap()

            return

        # register each location in the saliency map
        assert isinstance(e.data, Node)
        f = e.data

        def try_add_loc(n: Node) -> None:
            if isinstance(n, ElementPoint):
                assert self.saliency_map is not None
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
