"""Generates Features for things that aren't like their neighbors"""

from dataclasses import dataclass

from ..component import register_component
from ..location import Point
from ..perception import (
    FeatureExtractor,
    FeatureNode,
    PerceptionEvent,
    PointFeature,
    Settled,
    VisionData,
)
from .single import SingleFeature


class ColorNode(FeatureNode):
    type: int

    @property
    def attr_strs(self) -> list[str]:
        color: str

        # https://github.com/NetHack/NetHack/blob/8bb764e624aa228ce2a5374739408ed81b77d40e/include/color.h#L14
        match self.type:
            case 0:
                color = "BLACK"
            case 1:
                color = "RED"
            case 2:
                color = "GREEN"
            case 3:
                color = "BROWN"
            case 4:
                color = "BLUE"
            case 5:
                color = "MAGENTA"
            case 6:
                color = "CYAN"
            case 7:
                color = "GREY"
            case 8:
                color = "NO COLOR"
            case 9:
                color = "ORANGE"
            case 10:
                color = "BRIGHT GREEN"
            case 11:
                color = "YELLOW"
            case 12:
                color = "BRIGHT BLUE"
            case 13:
                color = "BRIGHT MAGENTA"
            case 14:
                color = "BRIGHT CYAN"
            case 15:
                color = "WHITE"
            case 16:
                color = "MAX"
            case _:
                raise Exception("impossible color")

        return [color]


# define CLR_BLACK 0
# define CLR_RED 1
# define CLR_GREEN 2
# define CLR_BROWN 3 /* on IBM, low-intensity yellow is brown */
# define CLR_BLUE 4
# define CLR_MAGENTA 5
# define CLR_CYAN 6
# define CLR_GRAY 7 /* low-intensity white */
# define NO_COLOR 8
# define CLR_ORANGE 9
# define CLR_BRIGHT_GREEN 10
# define CLR_YELLOW 11
# define CLR_BRIGHT_BLUE 12
# define CLR_BRIGHT_MAGENTA 13
# define CLR_BRIGHT_CYAN 14
# define CLR_WHITE 15
# define CLR_MAX 16


@dataclass(kw_only=True)
class ColorFeature(PointFeature[ColorNode]):
    """The color of a single feature."""

    feature_name: str = "Color"

    def _create_nodes(self) -> ColorNode:
        return ColorNode(type=self.type)

    def _dbfetch_nodes(self) -> ColorNode | None:
        return ColorNode.find_one("src.type = $type", params={"type": self.type})


@register_component("color", "perception")
class Color(FeatureExtractor[Point]):
    """A component for simulating the color of features based on the character
    value.
    """

    def __init__(self) -> None:
        super().__init__()
        self.queue: list[SingleFeature] = list()
        self.vd: VisionData | None = None
        self.single_settled = False

    def event_filter(self, e: PerceptionEvent) -> bool:
        """Filters out non-SingleFeatures and non-VisionData

        Args:
            e (PerceptionEvent): Any event on the perception bus

        Returns:
            bool: Returns True if the event is a SingleFeature or VisionData to
            keep processing it, False otherwise.
        """
        return isinstance(e.data, VisionData) or e.src_id.name == "single"

    def get_feature(self, e: PerceptionEvent) -> None:
        """Emits the color features.

        Args:
            e (PerceptionEvent): The VisionData or SingleFeature
        """
        if isinstance(e.data, SingleFeature):
            self.queue.append(e.data)
            return None

        if isinstance(e.data, Settled):
            self.single_settled = True

        if isinstance(e.data, VisionData):
            self.vd = e.data

        if self.single_settled and self.vd is not None:
            for s in self.queue:
                x, y = s.point
                self.pb_conn.send(
                    ColorFeature(
                        origin_id=self.id,
                        point=(x, y),
                        type=self.vd.colors[y, x],
                    )
                )

            self.settled()
            self.queue.clear()
            self.single_settled = False
            self.vd = None

        return None
