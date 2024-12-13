# mypy: disable-error-code="no-untyped-def"

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.attention import SaliencyMap, VisionAttention, VisionAttentionData
from roc.component import Component
from roc.event import Event
from roc.feature_extractors.color import Color
from roc.feature_extractors.delta import Delta
from roc.feature_extractors.distance import Distance
from roc.feature_extractors.flood import Flood
from roc.feature_extractors.line import Line
from roc.feature_extractors.motion import Motion
from roc.feature_extractors.shape import Shape
from roc.feature_extractors.single import Single
from roc.location import IntGrid, XLoc, YLoc
from roc.perception import Feature, FeatureNode, VisionData


class TestSaliencyMap:
    @pytest.fixture()
    def feature_for_test(self, empty_components) -> type:
        @dataclass(kw_only=True)
        class FeatureForTest(Feature[FeatureNode]):
            origin_id: tuple[str, str] = ("foo", "bar")
            feature_name: str = "Test"

            def get_points(self) -> set[tuple[XLoc, YLoc]]:
                return set()

            def node_hash(self) -> int:
                return 0

            def _create_nodes(self) -> FeatureNode:
                return FeatureNode()

            def _dbfetch_nodes(self) -> FeatureNode | None:
                return None

        return FeatureForTest

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

    def test_add(self, feature_for_test) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        f = feature_for_test()
        sm.add_val(1, 2, f)

        val = sm.get_val(0, 0)
        assert isinstance(val, list)
        assert len(val) == 0

        val = sm.get_val(1, 2)
        assert isinstance(val, list)
        assert len(val) == 1
        assert val[0] is f

    def test_add_multiple(self, feature_for_test) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        f1 = feature_for_test()
        f2 = feature_for_test()
        sm.add_val(2, 2, f1)
        sm.add_val(2, 2, f2)

        val = sm.get_val(0, 0)
        assert isinstance(val, list)
        assert len(val) == 0

        val = sm.get_val(2, 2)
        assert isinstance(val, list)
        assert len(val) == 2
        assert f1 in val
        assert f2 in val

    def test_clear(self, feature_for_test) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        f = feature_for_test()
        sm.add_val(1, 2, f)

        val = sm.get_val(1, 2)
        assert isinstance(val, list)
        assert len(val) == 1
        assert val[0] is f

        sm.clear()

        val = sm.get_val(1, 2)
        assert isinstance(val, list)
        assert len(val) == 0

    def test_strength(self, feature_for_test) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        f = feature_for_test()

        assert sm.get_strength(0, 1) == 0

        sm.add_val(0, 0, f)
        assert sm.get_strength(0, 0) == 1

        sm.add_val(1, 1, f)
        sm.add_val(1, 1, f)
        assert sm.get_strength(1, 1) == 2

        sm.add_val(2, 2, f)
        sm.add_val(2, 2, f)
        sm.add_val(2, 2, f)
        assert sm.get_strength(2, 2) == 3

        assert sm.get_max_strength() == 3
        assert sm.get_max_strength() == 3

    def test_copy(self, feature_for_test) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
            ]
        )
        sm1 = SaliencyMap(g)
        f = feature_for_test()
        sm1.add_val(0, 0, f)

        sm2 = deepcopy(sm1)

        assert sm1 is not sm2
        assert sm1.shape == sm2.shape
        assert sm2.grid is not None
        assert sm1.grid is not sm2.grid
        assert sm2.grid[0, 0] == 32
        assert sm2.grid[2, 2] == 99
        assert isinstance(sm2[0, 0], list)
        assert sm1[0, 0] is not sm2[0, 0]
        assert sm2[0, 0][0] is f

    def test_report(self, empty_components) -> None:
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
        distance = Component.get("distance", "perception")
        assert isinstance(distance, Distance)
        color = Component.get("color", "perception")
        assert isinstance(color, Color)
        shape = Component.get("shape", "perception")
        assert isinstance(shape, Shape)
        s = StubComponent(
            input_bus=delta.pb_conn.attached_bus,
            output_bus=attention.att_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))

        assert s.output.call_count == 2

        # screen 0
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, VisionAttentionData)
        sm = e.data.saliency_map
        d = sm.feature_report()
        assert len(d.keys()) == 6
        assert d["Flood"] == 2
        assert d["Line"] == 106
        assert d["Distance"] == 78
        assert d["Single"] == 13
        assert d["Color"] == 13
        assert d["Shape"] == 13

        # screen 1
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, VisionAttentionData)
        sm = e.data.saliency_map
        d = sm.feature_report()
        assert len(d.keys()) == 8
        assert d["Flood"] == 1
        assert d["Line"] == 106
        assert d["Single"] == 13
        assert d["Distance"] == 78
        assert d["Color"] == 13
        assert d["Shape"] == 13
        assert d["Delta"] == 2
        assert d["Motion"] == 2

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

    def test_basic(self, empty_components) -> None:
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
        distance = Component.get("distance", "perception")
        assert isinstance(distance, Distance)
        color = Component.get("color", "perception")
        assert isinstance(color, Color)
        shape = Component.get("shape", "perception")
        assert isinstance(shape, Shape)
        s = StubComponent(
            input_bus=delta.pb_conn.attached_bus,
            output_bus=attention.att_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))

        assert s.output.call_count == 2

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, VisionAttentionData)
        df = pd.DataFrame(
            {
                "x": {
                    0: 15,
                    1: 15,
                    2: 15,
                    3: 15,
                    4: 16,
                    5: 16,
                    6: 17,
                    7: 18,
                    8: 18,
                    9: 19,
                    10: 19,
                    11: 19,
                    12: 19,
                },
                "y": {
                    0: 3,
                    1: 4,
                    2: 5,
                    3: 8,
                    4: 5,
                    5: 6,
                    6: 5,
                    7: 5,
                    8: 8,
                    9: 3,
                    10: 4,
                    11: 5,
                    12: 8,
                },
                "strength": {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                    3: 1.0,
                    4: 1.0,
                    5: 1.0,
                    6: 1.0,
                    7: 1.0,
                    8: 1.0,
                    9: 1.0,
                    10: 1.0,
                    11: 1.0,
                    12: 1.0,
                },
                "label": {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 2,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 3,
                    9: 1,
                    10: 1,
                    11: 1,
                    12: 3,
                },
            }
        )
        assert np.allclose(e.data.focus_points, df)

        # second event
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, VisionAttentionData)
        df = pd.DataFrame(
            {
                "x": {
                    0: 17,
                    1: 16,
                    2: 15,
                    3: 15,
                    4: 15,
                    5: 15,
                    6: 16,
                    7: 17,
                    8: 18,
                    9: 18,
                    10: 19,
                    11: 19,
                    12: 19,
                    13: 19,
                },
                "y": {
                    0: 6,
                    1: 6,
                    2: 3,
                    3: 4,
                    4: 5,
                    5: 8,
                    6: 5,
                    7: 5,
                    8: 5,
                    9: 8,
                    10: 3,
                    11: 4,
                    12: 5,
                    13: 8,
                },
                "strength": {
                    0: 1.0,
                    1: 0.5967741935483871,
                    2: 0.4032258064516129,
                    3: 0.4032258064516129,
                    4: 0.4032258064516129,
                    5: 0.4032258064516129,
                    6: 0.4032258064516129,
                    7: 0.4032258064516129,
                    8: 0.4032258064516129,
                    9: 0.4032258064516129,
                    10: 0.4032258064516129,
                    11: 0.4032258064516129,
                    12: 0.4032258064516129,
                    13: 0.4032258064516129,
                },
                "label": {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 2,
                    6: 1,
                    7: 1,
                    8: 1,
                    9: 3,
                    10: 1,
                    11: 1,
                    12: 1,
                    13: 3,
                },
            }
        )
        assert np.allclose(e.data.focus_points, df)

    def test_four_screen(self, empty_components) -> None:
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
        distance = Component.get("distance", "perception")
        assert isinstance(distance, Distance)
        color = Component.get("color", "perception")
        assert isinstance(color, Color)
        shape = Component.get("shape", "perception")
        assert isinstance(shape, Shape)
        s = StubComponent(
            input_bus=delta.pb_conn.attached_bus,
            output_bus=attention.att_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))
        s.input_conn.send(VisionData.from_dict(screens[4]))
        s.input_conn.send(VisionData.from_dict(screens[6]))

        assert s.output.call_count == 4

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, VisionAttentionData)
        df = pd.DataFrame(
            {
                "x": {
                    0: 15,
                    1: 15,
                    2: 15,
                    3: 15,
                    4: 16,
                    5: 16,
                    6: 17,
                    7: 18,
                    8: 18,
                    9: 19,
                    10: 19,
                    11: 19,
                    12: 19,
                },
                "y": {
                    0: 3,
                    1: 4,
                    2: 5,
                    3: 8,
                    4: 5,
                    5: 6,
                    6: 5,
                    7: 5,
                    8: 8,
                    9: 3,
                    10: 4,
                    11: 5,
                    12: 8,
                },
                "strength": {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                    3: 1.0,
                    4: 1.0,
                    5: 1.0,
                    6: 1.0,
                    7: 1.0,
                    8: 1.0,
                    9: 1.0,
                    10: 1.0,
                    11: 1.0,
                    12: 1.0,
                },
                "label": {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 2,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 3,
                    9: 1,
                    10: 1,
                    11: 1,
                    12: 3,
                },
            }
        )
        assert np.allclose(e.data.focus_points, df)

        # second event
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, VisionAttentionData)
        df = pd.DataFrame(
            {
                "x": {
                    0: 17,
                    1: 16,
                    2: 15,
                    3: 15,
                    4: 15,
                    5: 15,
                    6: 16,
                    7: 17,
                    8: 18,
                    9: 18,
                    10: 19,
                    11: 19,
                    12: 19,
                    13: 19,
                },
                "y": {
                    0: 6,
                    1: 6,
                    2: 3,
                    3: 4,
                    4: 5,
                    5: 8,
                    6: 5,
                    7: 5,
                    8: 5,
                    9: 8,
                    10: 3,
                    11: 4,
                    12: 5,
                    13: 8,
                },
                "strength": {
                    0: 1.0,
                    1: 0.5967741935483871,
                    2: 0.4032258064516129,
                    3: 0.4032258064516129,
                    4: 0.4032258064516129,
                    5: 0.4032258064516129,
                    6: 0.4032258064516129,
                    7: 0.4032258064516129,
                    8: 0.4032258064516129,
                    9: 0.4032258064516129,
                    10: 0.4032258064516129,
                    11: 0.4032258064516129,
                    12: 0.4032258064516129,
                    13: 0.4032258064516129,
                },
                "label": {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 2,
                    6: 1,
                    7: 1,
                    8: 1,
                    9: 3,
                    10: 1,
                    11: 1,
                    12: 1,
                    13: 3,
                },
            }
        )
        assert np.allclose(e.data.focus_points, df)

        # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, VisionAttentionData)
        df = pd.DataFrame(
            {
                "x": {
                    0: 18,
                    1: 15,
                    2: 15,
                    3: 15,
                    4: 15,
                    5: 16,
                    6: 17,
                    7: 18,
                    8: 19,
                    9: 19,
                    10: 19,
                    11: 19,
                    12: 17,
                },
                "y": {
                    0: 5,
                    1: 3,
                    2: 4,
                    3: 5,
                    4: 8,
                    5: 5,
                    6: 5,
                    7: 8,
                    8: 3,
                    9: 4,
                    10: 5,
                    11: 8,
                    12: 6,
                },
                "strength": {
                    0: 1.0,
                    1: 0.39344262295081966,
                    2: 0.39344262295081966,
                    3: 0.39344262295081966,
                    4: 0.39344262295081966,
                    5: 0.39344262295081966,
                    6: 0.39344262295081966,
                    7: 0.39344262295081966,
                    8: 0.39344262295081966,
                    9: 0.39344262295081966,
                    10: 0.39344262295081966,
                    11: 0.39344262295081966,
                    12: 0.2786885245901639,
                },
                "label": {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 2,
                    5: 1,
                    6: 1,
                    7: 3,
                    8: 1,
                    9: 1,
                    10: 1,
                    11: 3,
                    12: 1,
                },
            }
        )
        assert np.allclose(e.data.focus_points, df)

        # fourth event
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, VisionAttentionData)
        df = pd.DataFrame(
            {
                "x": {
                    0: 18,
                    1: 18,
                    2: 15,
                    3: 15,
                    4: 15,
                    5: 15,
                    6: 16,
                    7: 17,
                    8: 18,
                    9: 19,
                    10: 19,
                    11: 19,
                    12: 19,
                },
                "y": {
                    0: 6,
                    1: 5,
                    2: 3,
                    3: 4,
                    4: 5,
                    5: 8,
                    6: 5,
                    7: 5,
                    8: 8,
                    9: 3,
                    10: 4,
                    11: 5,
                    12: 8,
                },
                "strength": {
                    0: 1.0,
                    1: 0.6612903225806451,
                    2: 0.4032258064516129,
                    3: 0.4032258064516129,
                    4: 0.4032258064516129,
                    5: 0.4032258064516129,
                    6: 0.4032258064516129,
                    7: 0.4032258064516129,
                    8: 0.4032258064516129,
                    9: 0.4032258064516129,
                    10: 0.4032258064516129,
                    11: 0.4032258064516129,
                    12: 0.4032258064516129,
                },
                "label": {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 2,
                    6: 1,
                    7: 1,
                    8: 3,
                    9: 1,
                    10: 1,
                    11: 1,
                    12: 3,
                },
            }
        )
        assert np.allclose(e.data.focus_points, df)
