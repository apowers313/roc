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
import pandas as pd
from scipy.ndimage import label
from skimage.morphology import reconstruction
from strictly_typed_pandas import DataSet

from .component import Component, register_component
from .config import Config
from .event import Event, EventBus
from .location import DebugGrid, Grid, IntGrid
from .perception import (
    Feature,
    FeatureExtractor,
    Perception,
    PerceptionEvent,
    Settled,
    VisionData,
)
from .reporting.observability import Observability


class VisionAttentionSchema:
    x: int
    y: int
    strength: float
    label: int


@dataclass
class VisionAttentionData:
    focus_points: DataSet[VisionAttentionSchema]
    saliency_map: SaliencyMap

    def __str__(self) -> str:
        assert self.saliency_map.grid is not None
        dg = DebugGrid(self.saliency_map.grid)

        for idx, row in self.focus_points.iterrows():
            x = int(row["x"])
            y = int(row["y"])
            dg.set_style(x, y, back_brightness=row["strength"], back_hue=1)

        return f"{str(dg)}\n\nFocus Points:\n{self.focus_points}"


AttentionData = VisionAttentionData
AttentionEvent = Event[AttentionData]


class Attention(Component, ABC):
    bus = EventBus[AttentionData]("attention")


class SaliencyMap(Grid[list[Feature[Any]]]):
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

    def __str__(self) -> str:
        return str(self.to_debug_grid())

    def to_debug_grid(self) -> DebugGrid:
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

        return dg

    def to_html_vals(self) -> dict[str, list[list[str | int]]]:
        dg = self.to_debug_grid()
        return dg.to_html_vals()

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

    def add_val(self, x: int, y: int, val: Feature[Any]) -> None:
        feature_list = self.get_val(x, y)
        feature_list.append(val)

    def get_max_strength(self) -> int:
        max = 0
        for y in range(self.height):
            for x in range(self.width):
                curr = self.get_strength(x, y)
                if max < curr:
                    max = curr

        return max

    def get_strength(self, x: int, y: int) -> int:
        feature_list = self.get_val(x, y)
        # TODO: not really sure that the strength should depend on the number of features
        ret = len(feature_list)

        def add_strength(f: Feature[Any]) -> None:
            nonlocal ret

            # TODO: this is pretty arbitrary and might be biased based on my
            # domain knowledge... I suspect I will come back and modify this
            # based on object recognition and other factors at some point in
            # the future
            if f.feature_name == "Single":
                ret += 10
            if f.feature_name == "Delta":
                ret += 15
            if f.feature_name == "Motion":
                ret += 20

        for f in feature_list:
            add_strength(f)

        return ret

    def feature_report(self) -> dict[str, int]:
        feature_id: dict[str, set[int]] = dict()

        # create a set of unique IDs for every distinct feature
        for row, col in np.ndindex(self.shape):
            feature_list = self[row, col]
            for f in feature_list:
                feature_name = f.feature_name
                if feature_name not in feature_id:
                    feature_id[feature_name] = set()
                feature_id[feature_name].add(id(f))

        # count all the sets
        ret = {k: len(feature_id[k]) for k in feature_id}
        return ret

    def get_focus(self) -> DataSet[VisionAttentionSchema]:
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

        # find peaks through dilation
        seed = np.copy(fkimg)
        seed[1:-1, 1:-1] = fkimg.min()
        rec = reconstruction(seed, fkimg, method="dilation")
        peaks = fkimg - rec

        # get coordinates of peaks
        nz = peaks.nonzero()
        coords = np.column_stack(nz)

        # label points that are adjacent / diagonal
        structure = np.ones((3, 3), dtype=int)
        labeled, ncomponents = label(peaks, structure)

        # get values for each coordinate
        flat_indicies = np.ravel_multi_index(tuple(coords.T), fkimg.shape)
        vals = np.take(fkimg, flat_indicies)
        labels = np.take(labeled, flat_indicies)

        # create table of peak info, ordered by strength
        df = (
            pd.DataFrame(
                {
                    "x": nz[0],
                    "y": nz[1],
                    "strength": vals,
                    "label": labels,
                }
            )
            .astype({"x": int, "y": int, "strength": float, "label": int})
            .sort_values("strength", ascending=False)
            .reset_index(drop=True)
        )

        return DataSet[VisionAttentionSchema](df)


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

    @Observability.tracer.start_as_current_span("do_attention")
    def do_attention(self, e: PerceptionEvent) -> None:
        # create right-sized SaliencyMap based on VisionData
        if isinstance(e.data, VisionData):
            self.saliency_map.grid = IntGrid(e.data.chars)
            return

        # check to see if all feature extractors have settled
        if isinstance(e.data, Settled):
            self.settled.add(str(e.src_id))

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

                # reset
                self.settled.clear()
                self.saliency_map = SaliencyMap()

            return

        # register each location in the saliency map
        assert isinstance(e.data, Feature)
        f = e.data

        # create saliency map
        for p in f.get_points():
            self.saliency_map.add_val(p[0], p[1], f)


# TODO: other attention classes


class CrossModalAttention(Attention):
    # TODO: listen for attention events
    # TODO: select and emit a single event
    pass
