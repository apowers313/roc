from roc.component import Component


def test_component_exists():
    c = Component("myname", "mytype")
    assert c.name == "myname"
    assert c.type == "mytype"
