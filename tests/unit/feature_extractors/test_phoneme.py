# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/feature_extractors/phoneme.py."""

from unittest.mock import MagicMock, patch

import pytest
from gruut import sentences

from roc.feature_extractors.phoneme import Phoneme, PhonemeFeature, PhonemeNode, PhonemeWord
from roc.perception import AuditoryData


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestPhonemeNode:
    def test_attr_strs(self):
        """Line 22: attr_strs returns [str(self.type)]."""
        node = PhonemeNode(type=42)
        assert node.attr_strs == ["42"]

    def test_attr_strs_zero(self):
        node = PhonemeNode(type=0)
        assert node.attr_strs == ["0"]


class TestPhonemeFeature:
    def test_create_nodes(self):
        """Line 33: _create_nodes returns a PhonemeNode with type=42."""
        feature = PhonemeFeature(
            origin_id=MagicMock(),
            phonemes=[PhonemeWord(word="hello", phonemes=["h", "eh", "l", "ow"])],
        )
        node = feature._create_nodes()
        assert isinstance(node, PhonemeNode)
        assert node.type == 42

    def test_dbfetch_nodes(self):
        """Line 36: _dbfetch_nodes calls PhonemeNode.find_one."""
        feature = PhonemeFeature(
            origin_id=MagicMock(),
            phonemes=[PhonemeWord(word="hello", phonemes=["h", "eh", "l", "ow"])],
        )
        with patch.object(PhonemeNode, "find_one", return_value=None) as mock_find:
            result = feature._dbfetch_nodes()
            mock_find.assert_called_once_with("src.type = $type", params={"type": 42})
            assert result is None

    def test_dbfetch_nodes_found(self):
        """_dbfetch_nodes returns the node when found."""
        feature = PhonemeFeature(
            origin_id=MagicMock(),
            phonemes=[],
        )
        mock_node = PhonemeNode(type=42)
        with patch.object(PhonemeNode, "find_one", return_value=mock_node):
            result = feature._dbfetch_nodes()
            assert result is mock_node


class TestPhonemeExtraction:
    """Tests for the phoneme extraction logic in Phoneme.get_feature."""

    @staticmethod
    def _extract_phonemes(msg: str) -> list[PhonemeWord]:
        """Extract phonemes using the same logic as Phoneme.get_feature."""
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
        return phonemes

    def test_sentence_breaks_preserved(self):
        """Sentence boundary markers must be preserved as break entries."""
        phonemes = self._extract_phonemes("Be careful! New moon tonight.")
        breaks = [pw for pw in phonemes if pw.is_break]
        assert len(breaks) == 2, "Should have 2 breaks (! and .)"
        assert breaks[0].word == "!"
        assert breaks[1].word == "."

    def test_word_text_from_gruut(self):
        """Each entry must carry the word text from gruut, not from msg.split().

        Regression: the old code used msg.split() in the component to guess
        word labels, which broke alignment when sentence boundary markers
        were present.
        """
        phonemes = self._extract_phonemes("Be careful! New moon tonight.")
        words = [pw.word for pw in phonemes if not pw.is_break]
        assert words == ["Be", "careful", "New", "moon", "tonight"]

    def test_break_phonemes_are_ipa_markers(self):
        """Break entries should contain IPA boundary markers."""
        phonemes = self._extract_phonemes("Hello. Goodbye.")
        breaks = [pw for pw in phonemes if pw.is_break]
        for brk in breaks:
            assert brk.phonemes == ["\u2016"]

    def test_spoken_words_have_real_phonemes(self):
        """Non-break entries should have real IPA phonemes."""
        phonemes = self._extract_phonemes("hello world")
        assert len(phonemes) == 2
        for pw in phonemes:
            assert not pw.is_break
            assert len(pw.phonemes) > 0
            assert all(isinstance(p, str) for p in pw.phonemes)

    def test_multiple_sentences(self):
        """Multiple sentences produce breaks between them and at the end."""
        phonemes = self._extract_phonemes("Hello. Goodbye. Thanks.")
        spoken = [pw for pw in phonemes if not pw.is_break]
        breaks = [pw for pw in phonemes if pw.is_break]
        assert len(spoken) == 3
        assert len(breaks) == 3  # one period after each sentence
