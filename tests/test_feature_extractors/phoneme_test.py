# mypy: disable-error-code="no-untyped-def"


from helpers.util import StubComponent

from roc.component import Component
from roc.feature_extractors.phoneme import Phoneme, PhonemeFeature
from roc.perception import AuditoryData, Settled


class TestSingle:
    def test_phoneme_exists(self) -> None:
        Phoneme()

    def test_basic(self, empty_components) -> None:
        c = Component.get("phoneme", "perception")
        assert isinstance(c, Phoneme)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(AuditoryData("You cannot eat that!"))

        assert s.output.call_count == 2

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e.data, PhonemeFeature)
        pf = e.data
        assert pf.phonemes == [
            ["j", "ˈu"],
            ["k", "æ", "n", "ˈɑ", "t"],
            ["ˈi", "t"],
            ["ð", "ˈæ", "t"],
            ["‖"],
        ]

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e.data, Settled)
