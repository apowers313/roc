"""Generates Features for things that aren't like their neighbors"""

from dataclasses import dataclass

from gruut import sentences

from ..location import Point
from ..perception import (
    AuditoryData,
    Feature,
    FeatureExtractor,
    FeatureNode,
    PerceptionEvent,
)


class PhonemeNode(FeatureNode):
    type: int

    @property
    def attr_strs(self) -> list[str]:
        return [str(self.type)]


@dataclass(kw_only=True)
class PhonemeFeature(Feature):
    """A collection of phoneme nodes."""

    feature_name: str = "Phonemes"
    phonemes: list[list[str]]

    def _create_nodes(self) -> PhonemeNode:
        return PhonemeNode(type=42)  # XXX TODO type

    def _dbfetch_nodes(self) -> PhonemeNode | None:
        return PhonemeNode.find_one("src.type = $type", params={"type": 42})  # XXX TODO type


class Phoneme(FeatureExtractor[Point]):
    """A component for creating phonemes from auditory events"""

    name: str = "phoneme"
    type: str = "perception"

    def event_filter(self, e: PerceptionEvent) -> bool:
        """Filters out non-AuditoryData

        Args:
            e (PerceptionEvent): Any event on the perception bus

        Returns:
            bool: Returns True if the event is AuditoryData to keep processing it,
            False otherwise.
        """
        return isinstance(e.data, AuditoryData)

    def get_feature(self, e: PerceptionEvent) -> None:
        """Emits phonemes of an auditory event.

        Args:
            e (PerceptionEvent): The AuditoryData
        """
        ad = e.data
        assert isinstance(ad, AuditoryData)

        phonemes: list[list[str]] = []
        for sent in sentences(ad.msg, lang="en-us"):
            for word in sent:
                if word.phonemes:
                    phonemes.append(word.phonemes)

        self.pb_conn.send(PhonemeFeature(origin_id=self.id, phonemes=phonemes))
        self.settled()
