# mypy: disable-error-code="no-untyped-def"


from copy import deepcopy

import numpy as np
from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.attention import SaliencyMap, VisionAttention
from roc.component import Component
from roc.feature_extractors.delta import Delta
from roc.feature_extractors.flood import Flood
from roc.feature_extractors.line import Line
from roc.feature_extractors.motion import Motion
from roc.feature_extractors.single import Single
from roc.graphdb import Node
from roc.location import IntGrid
from roc.perception import VisionData


class TestSaliencyMap:
    def test_exists(self) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        SaliencyMap(g)

    def test_get(self) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        val = sm.get_val(0, 0)
        assert isinstance(val, list)
        assert len(val) == 0

        val = sm.get_val(2, 3)
        assert isinstance(val, list)
        assert len(val) == 0

    def test_add(self) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        n = Node(labels=["TestNode"])
        sm.add_val(1, 2, n)

        val = sm.get_val(0, 0)
        assert isinstance(val, list)
        assert len(val) == 0

        val = sm.get_val(1, 2)
        assert isinstance(val, list)
        assert len(val) == 1
        assert val[0] is n

    def test_add_multiple(self) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        n1 = Node(labels=["TestNode"])
        n2 = Node(labels=["TestNode"])
        sm.add_val(2, 2, n1)
        sm.add_val(2, 2, n2)

        val = sm.get_val(0, 0)
        assert isinstance(val, list)
        assert len(val) == 0

        val = sm.get_val(2, 2)
        assert isinstance(val, list)
        assert len(val) == 2
        assert n1 in val
        assert n2 in val

    def test_clear(self) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        n = Node(labels=["TestNode"])
        sm.add_val(1, 2, n)

        val = sm.get_val(1, 2)
        assert isinstance(val, list)
        assert len(val) == 1
        assert val[0] is n

        sm.clear()

        val = sm.get_val(1, 2)
        assert isinstance(val, list)
        assert len(val) == 0

    def test_strength(self) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        n = Node(labels=["TestNode"])

        assert sm.get_strength(0, 1) == 0

        sm.add_val(0, 0, n)
        assert sm.get_strength(0, 0) == 1

        sm.add_val(1, 1, n)
        sm.add_val(1, 1, n)
        assert sm.get_strength(1, 1) == 2

        sm.add_val(2, 2, n)
        sm.add_val(2, 2, n)
        sm.add_val(2, 2, n)
        assert sm.get_strength(2, 2) == 3

        assert sm.get_max_strength() == 3
        assert sm.get_max_strength() == 3

    def test_copy(self) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm1 = SaliencyMap(g)
        sm2 = deepcopy(sm1)

        assert id(sm1) != id(sm2)
        # assert id(sm1.grid) != id(sm2.grid)
        arr1 = np.array([1, 2, 3])
        arr2 = deepcopy(arr1)
        assert id(arr1) != id(arr2)

    # def test_str(self) -> None:
    #     g = Grid(
    #         [
    #             [32, 32, 32],
    #             [49, 50, 51],
    #             [97, 98, 99],
    #         ]
    #     )
    #     sm = SaliencyMap(g)
    #     n = Node(labels=["TestNode"])
    #     sm.add_val(0, 0, n)
    #     sm.add_val(1, 1, n)
    #     sm.add_val(1, 1, n)
    #     sm.add_val(2, 2, n)
    #     sm.add_val(2, 2, n)
    #     sm.add_val(2, 2, n)

    #     assert str(sm) == "\u2591  \n \u2593 \n  \u2588\n"


class TestVisionAttention:
    def test_exists(self) -> None:
        VisionAttention()

    def test_basic(self, empty_components, memory_profile) -> None:
        delta = Component.get("delta", "perception")
        assert isinstance(delta, Delta)
        attention = Component.get("vision", "attention")
        assert isinstance(attention, VisionAttention)
        flood = Component.get("flood", "perception")
        assert isinstance(flood, Flood)
        line = Component.get("line", "perception")
        assert isinstance(line, Line)
        motion = Component.get("motion", "perception")
        assert isinstance(motion, Motion)
        single = Component.get("single", "perception")
        assert isinstance(single, Single)
        s = StubComponent(
            input_bus=delta.pb_conn.attached_bus,
            output_bus=attention.att_conn.attached_bus,
        )

        # print("grid1", VisionData(screens[0]["chars"]))
        # print("grid2", VisionData(screens[1]["chars"]))
        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))

        assert attention.saliency_map is not None
        assert attention.saliency_map.grid is not None
        print("saliency features", attention.saliency_map.size)  # noqa: T201
        print("vision:\n", IntGrid(screens[0]["chars"]))  # noqa: T201
        print(f"saliency map:\n{attention.saliency_map}")  # noqa: T201
        print("saliency max strength", attention.saliency_map.get_max_strength())  # noqa: T201
        print("saliency strength (0,0)", attention.saliency_map.get_strength(0, 0))  # noqa: T201
        print("saliency strength (16,5)", attention.saliency_map.get_strength(16, 5))  # noqa: T201
        print("saliency strength (16,6)", attention.saliency_map.get_strength(16, 6))  # noqa: T201
        print(attention.saliency_map.grid.get_point(16, 6))  # noqa: T201
        print(attention.saliency_map.grid.get_point(17, 6))  # noqa: T201

        # assert s.output.call_count == 2

        # # first event
        # e = s.output.call_args_list[0].args[0]
        # assert isinstance(e, Event)

        # # second event
        # e = s.output.call_args_list[1].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, Settled)
