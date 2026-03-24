"""Generates Features for things that aren't like their neighbors"""

from dataclasses import dataclass

from gruut import sentences

from ..graphdb import FindQueryOpts

from ..location import Point
from ..perception import (
    AuditoryData,
    Feature,
    FeatureExtractor,
    FeatureNode,
    PerceptionEvent,
)


class PhonemeNode(FeatureNode):
    """Graph node representing a phoneme feature."""

    type: int

    @property
    def attr_strs(self) -> list[str]:
        """Returns the type as a string."""
        return [str(self.type)]


@dataclass(kw_only=True)
class PhonemeWord:
    """A single word's phoneme decomposition from gruut."""

    word: str
    phonemes: list[str]
    is_break: bool = False


@dataclass(kw_only=True)
class PhonemeFeature(Feature):
    """A collection of phoneme nodes."""

    feature_name: str = "Phonemes"
    phonemes: list[PhonemeWord]

    def _create_nodes(self) -> PhonemeNode:
        """Creates a new PhonemeNode."""
        return PhonemeNode(type=42)  # XXX TODO type

    def _dbfetch_nodes(self) -> PhonemeNode | None:
        """Looks up an existing PhonemeNode."""
        return PhonemeNode.find_one(
            "src.type = $type", query_opts=FindQueryOpts(params={"type": 42})
        )  # XXX TODO type


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

        msg = ad.msg.strip("\x00").strip()
        if not msg:
            self.settled()
            return

        phonemes: list[PhonemeWord] = []
        for sent in sentences(msg, lang="en-us"):
            for word in sent:
                if word.phonemes:
                    phonemes.append(
                        PhonemeWord(
                            word=word.text,
                            phonemes=list(word.phonemes),
                            is_break=word.is_major_break or word.is_minor_break,
                        )
                    )

        self.pb_conn.send(PhonemeFeature(origin_id=self.id, phonemes=phonemes))
        self.settled()
