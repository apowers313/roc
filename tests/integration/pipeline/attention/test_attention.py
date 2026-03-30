# mypy: disable-error-code="no-untyped-def"
from __future__ import annotations

"""Integration tests for VisionAttention and SaliencyMap.feature_report.

These tests wire up real perception components and send real NetHack screen
data through the pipeline.
"""

from typing import Any

import numpy as np
import pandas as pd
from helpers.nethack_screens import screens
from helpers.nethack_screens2 import screens as screens2
from helpers.util import StubComponent

from roc.pipeline.attention.attention import AttentionSettled, VisionAttention, VisionAttentionData
from roc.framework.config import Config
from roc.framework.component import Component
from roc.framework.event import Event
from roc.perception.feature_extractors.delta import Delta
from roc.perception.base import VisionData


def _make_attention_stub(empty_components: None) -> StubComponent[Any, Any]:
    """Set up perception components and return a wired StubComponent."""
    Config.get().attention_cycles = 1
    delta = Component.get("delta", "perception")
    assert isinstance(delta, Delta)
    attention = Component.get("vision", "attention")
    assert isinstance(attention, VisionAttention)
    assert Component.get("flood", "perception") is not None
    assert Component.get("line", "perception") is not None
    assert Component.get("motion", "perception") is not None
    assert Component.get("single", "perception") is not None
    assert Component.get("distance", "perception") is not None
    assert Component.get("color", "perception") is not None
    assert Component.get("shape", "perception") is not None
    return StubComponent(
        input_bus=delta.pb_conn.attached_bus,
        output_bus=attention.att_conn.attached_bus,
        filter=lambda e: not isinstance(e.data, AttentionSettled),
    )


def _assert_focus_points_event(
    s: StubComponent[Any, Any], call_index: int, df: pd.DataFrame
) -> None:
    """Assert that the event at call_index has focus_points matching df."""
    e = s.output.call_args_list[call_index].args[0]
    assert isinstance(e, Event)
    assert isinstance(e.data, VisionAttentionData)
    assert np.allclose(e.data.focus_points, df)


def _screens_first_event_df() -> pd.DataFrame:
    """Expected focus_points DataFrame for screens[0]."""
    return pd.DataFrame(
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
            "y": {0: 3, 1: 4, 2: 5, 3: 8, 4: 5, 5: 6, 6: 5, 7: 5, 8: 8, 9: 3, 10: 4, 11: 5, 12: 8},
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


def _screens_second_event_df() -> pd.DataFrame:
    """Expected focus_points DataFrame for screens[1]."""
    return pd.DataFrame(
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
                2: 4,
                3: 5,
                4: 8,
                5: 3,
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
                1: 0.6129032258064516,
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
                4: 2,
                5: 1,
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


class TestSaliencyMapReport:
    def test_report(self, empty_components) -> None:
        s = _make_attention_stub(empty_components)

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
        assert d["Flood"] == 2
        assert d["Line"] == 106
        assert d["Single"] == 13
        assert d["Distance"] == 78
        assert d["Color"] == 13
        assert d["Shape"] == 13
        assert d["Delta"] == 2
        assert d["Motion"] == 2


class TestVisionAttention:
    def test_exists(self) -> None:
        VisionAttention()

    def test_basic(self, empty_components) -> None:
        s = _make_attention_stub(empty_components)

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))

        assert s.output.call_count == 2

        _assert_focus_points_event(s, 0, _screens_first_event_df())
        _assert_focus_points_event(s, 1, _screens_second_event_df())

    def test_four_screen(self, empty_components) -> None:
        s = _make_attention_stub(empty_components)

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))
        s.input_conn.send(VisionData.from_dict(screens[4]))
        s.input_conn.send(VisionData.from_dict(screens[6]))

        assert s.output.call_count == 4

        _assert_focus_points_event(s, 0, _screens_first_event_df())
        _assert_focus_points_event(s, 1, _screens_second_event_df())

        # third event
        _assert_focus_points_event(
            s,
            2,
            pd.DataFrame(
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
                        1: 4,
                        2: 5,
                        3: 8,
                        4: 3,
                        5: 5,
                        6: 5,
                        7: 8,
                        8: 3,
                        9: 5,
                        10: 4,
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
                        3: 2,
                        4: 1,
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
            ),
        )

        # fourth event
        _assert_focus_points_event(
            s,
            3,
            pd.DataFrame(
                {
                    "x": {
                        0: 18,
                        1: 18,
                        2: 15,
                        3: 15,
                        4: 15,
                        5: 16,
                        6: 15,
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
                        2: 5,
                        3: 4,
                        4: 3,
                        5: 5,
                        6: 8,
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
                        5: 1,
                        6: 2,
                        7: 1,
                        8: 3,
                        9: 1,
                        10: 1,
                        11: 1,
                        12: 3,
                    },
                }
            ),
        )

    def test_four_screen2(self, empty_components) -> None:
        s = _make_attention_stub(empty_components)

        s.input_conn.send(VisionData.from_dict(screens2[0]))
        s.input_conn.send(VisionData.from_dict(screens2[1]))
        s.input_conn.send(VisionData.from_dict(screens2[3]))
        s.input_conn.send(VisionData.from_dict(screens2[4]))

        assert s.output.call_count == 4

        # first event
        _assert_focus_points_event(
            s,
            0,
            pd.DataFrame(
                {
                    "x": {
                        0: 1,
                        1: 1,
                        2: 2,
                        3: 2,
                        4: 2,
                        5: 3,
                        6: 4,
                        7: 4,
                        8: 5,
                        9: 5,
                        10: 6,
                        11: 7,
                        12: 7,
                        13: 7,
                    },
                    "y": {
                        0: 11,
                        1: 15,
                        2: 11,
                        3: 12,
                        4: 13,
                        5: 11,
                        6: 11,
                        7: 13,
                        8: 11,
                        9: 14,
                        10: 11,
                        11: 11,
                        12: 14,
                        13: 15,
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
                        13: 1.0,
                    },
                    "label": {
                        0: 1,
                        1: 2,
                        2: 1,
                        3: 1,
                        4: 1,
                        5: 1,
                        6: 1,
                        7: 3,
                        8: 1,
                        9: 3,
                        10: 1,
                        11: 1,
                        12: 4,
                        13: 4,
                    },
                }
            ),
        )

        # second event
        _assert_focus_points_event(
            s,
            1,
            pd.DataFrame(
                {
                    "x": {
                        0: 4,
                        1: 5,
                        2: 1,
                        3: 2,
                        4: 2,
                        5: 1,
                        6: 2,
                        7: 4,
                        8: 3,
                        9: 4,
                        10: 5,
                        11: 6,
                        12: 7,
                        13: 7,
                        14: 7,
                    },
                    "y": {
                        0: 14,
                        1: 14,
                        2: 11,
                        3: 12,
                        4: 13,
                        5: 15,
                        6: 11,
                        7: 11,
                        8: 11,
                        9: 13,
                        10: 11,
                        11: 11,
                        12: 11,
                        13: 14,
                        14: 15,
                    },
                    "strength": {
                        0: 1.0,
                        1: 0.6031746031746031,
                        2: 0.4126984126984127,
                        3: 0.4126984126984127,
                        4: 0.4126984126984127,
                        5: 0.4126984126984127,
                        6: 0.4126984126984127,
                        7: 0.4126984126984127,
                        8: 0.4126984126984127,
                        9: 0.4126984126984127,
                        10: 0.4126984126984127,
                        11: 0.4126984126984127,
                        12: 0.4126984126984127,
                        13: 0.4126984126984127,
                        14: 0.4126984126984127,
                    },
                    "label": {
                        0: 3,
                        1: 3,
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
                        13: 4,
                        14: 4,
                    },
                }
            ),
        )

        # third event
        _assert_focus_points_event(
            s,
            2,
            pd.DataFrame(
                {
                    "x": {
                        0: 2,
                        1: 1,
                        2: 1,
                        3: 2,
                        4: 2,
                        5: 3,
                        6: 4,
                        7: 4,
                        8: 5,
                        9: 6,
                        10: 7,
                        11: 7,
                        12: 7,
                        13: 4,
                    },
                    "y": {
                        0: 12,
                        1: 11,
                        2: 15,
                        3: 11,
                        4: 13,
                        5: 11,
                        6: 11,
                        7: 13,
                        8: 11,
                        9: 11,
                        10: 14,
                        11: 11,
                        12: 15,
                        13: 14,
                    },
                    "strength": {
                        0: 1.0,
                        1: 0.6097560975609756,
                        2: 0.6097560975609756,
                        3: 0.6097560975609756,
                        4: 0.6097560975609756,
                        5: 0.6097560975609756,
                        6: 0.6097560975609756,
                        7: 0.6097560975609756,
                        8: 0.6097560975609756,
                        9: 0.6097560975609756,
                        10: 0.6097560975609756,
                        11: 0.6097560975609756,
                        12: 0.6097560975609756,
                        13: 0.43902439024390244,
                    },
                    "label": {
                        0: 1,
                        1: 1,
                        2: 2,
                        3: 1,
                        4: 1,
                        5: 1,
                        6: 1,
                        7: 3,
                        8: 1,
                        9: 1,
                        10: 4,
                        11: 1,
                        12: 4,
                        13: 3,
                    },
                }
            ),
        )

        # fourth event
        _assert_focus_points_event(
            s,
            3,
            pd.DataFrame(
                {
                    "x": {
                        0: 1,
                        1: 1,
                        2: 2,
                        3: 2,
                        4: 2,
                        5: 3,
                        6: 4,
                        7: 4,
                        8: 5,
                        9: 6,
                        10: 7,
                        11: 7,
                        12: 7,
                    },
                    "y": {
                        0: 11,
                        1: 15,
                        2: 11,
                        3: 12,
                        4: 13,
                        5: 11,
                        6: 11,
                        7: 13,
                        8: 11,
                        9: 11,
                        10: 11,
                        11: 14,
                        12: 15,
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
                        1: 2,
                        2: 1,
                        3: 1,
                        4: 1,
                        5: 1,
                        6: 1,
                        7: 3,
                        8: 1,
                        9: 1,
                        10: 1,
                        11: 4,
                        12: 4,
                    },
                }
            ),
        )
