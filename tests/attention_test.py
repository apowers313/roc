# mypy: disable-error-code="no-untyped-def"

import os

import psutil
from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.attention import SaliencyMap, VisionAttention, float_to_density
from roc.component import Component
from roc.feature_extractors.delta import Delta
from roc.feature_extractors.flood import Flood
from roc.feature_extractors.line import Line
from roc.feature_extractors.motion import Motion
from roc.feature_extractors.single import Single
from roc.graphdb import Node
from roc.perception import VisionData

screen0 = VisionData(screens[0]["chars"])
screen1 = VisionData(screens[1]["chars"])


class TestSaliencyMap:
    def test_exists(self) -> None:
        SaliencyMap(3, 3)

    def test_get(self) -> None:
        sm = SaliencyMap(3, 4)
        val = sm.get_val(0, 0)
        assert isinstance(val, list)
        assert len(val) == 0

        val = sm.get_val(2, 3)
        assert isinstance(val, list)
        assert len(val) == 0

    def test_add(self) -> None:
        sm = SaliencyMap(3, 3)
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
        sm = SaliencyMap(3, 3)
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
        sm = SaliencyMap(3, 3)
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
        sm = SaliencyMap(3, 3)
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

    def test_str(self) -> None:
        sm = SaliencyMap(3, 3)
        n = Node(labels=["TestNode"])
        sm.add_val(0, 0, n)
        sm.add_val(1, 1, n)
        sm.add_val(1, 1, n)
        sm.add_val(2, 2, n)
        sm.add_val(2, 2, n)
        sm.add_val(2, 2, n)

        assert str(sm) == "\u2591  \n \u2593 \n  \u2588\n"


class TestDensity:
    def test_density(self) -> None:
        EMPTY_CHR = 32  # ASCII space
        LIGHT_CHR = 9617  # Unicode 2591
        MED_CHR = 9618  # Unicode 2592
        DARK_CHR = 9619  # Unicode 2593
        FULL_CHR = 9608  # Unicode 2588
        shade_map: list[int] = [EMPTY_CHR, LIGHT_CHR, MED_CHR, DARK_CHR, FULL_CHR]

        assert float_to_density(0.00, shade_map) == EMPTY_CHR
        assert float_to_density(0.19, shade_map) == EMPTY_CHR
        assert float_to_density(0.21, shade_map) == LIGHT_CHR
        assert float_to_density(0.39, shade_map) == LIGHT_CHR
        assert float_to_density(0.41, shade_map) == MED_CHR
        assert float_to_density(0.59, shade_map) == MED_CHR
        assert float_to_density(0.61, shade_map) == DARK_CHR
        assert float_to_density(0.79, shade_map) == DARK_CHR
        assert float_to_density(0.81, shade_map) == FULL_CHR
        assert float_to_density(0.99, shade_map) == FULL_CHR
        assert float_to_density(1, shade_map) == FULL_CHR


class TestVisionAttention:
    def test_exists(self) -> None:
        VisionAttention()

    def test_basic(self, empty_components, memory_profile) -> None:
        # inner psutil function
        def process_memory():
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return mem_info.rss

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

        s.input_conn.send(VisionData(screens[0]["chars"]))
        s.input_conn.send(VisionData(screens[1]["chars"]))

        assert attention.saliency_map
        print("saliency features", attention.saliency_map.size)  # noqa: T201
        print("vision:\n", VisionData(screens[0]["chars"]))  # noqa: T201
        print("saliency map:\n", attention.saliency_map)  # noqa: T201
        # assert s.output.call_count == 2

        # # first event
        # e = s.output.call_args_list[0].args[0]
        # assert isinstance(e, Event)

        # # second event
        # e = s.output.call_args_list[1].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, Settled)
