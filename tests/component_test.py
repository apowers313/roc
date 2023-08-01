# mypy: disable-error-code="no-untyped-def"

from roc.component import Component


class TestComponent:
    def test_component_exists(self):
        c = Component("myname", "mytype")
        assert c.name == "myname"
        assert c.type == "mytype"
