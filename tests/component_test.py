# mypy: disable-error-code="no-untyped-def"

from roc.component import Component, component_registry, register_component

# class TestComponent:
#     def test_component_exists(self):
#         c = Component()
#         assert c.name == "myname"
#         assert c.type == "mytype"


class TestRegisterDecorator:
    def test_decorator(self):
        @register_component(name="foo", type="bar")
        class Foo(Component):
            pass

        assert len(component_registry) == 1
        assert "foo:bar" in component_registry

    def test_decorator_doc(self):
        @register_component("bar", "baz")
        class Bar(Component):
            """This is a Bar doc"""

        assert "bar:baz" in component_registry
        assert component_registry["bar:baz"].__doc__ == "This is a Bar doc"

    def test_decorator_creates_class(self):
        @register_component("bar", "baz")
        class Bar(Component):
            pass

        assert "bar:baz" in component_registry
        c = component_registry["bar:baz"]()
        assert isinstance(c, Component)
        assert isinstance(c, Bar)
        assert c.name == "bar"
        assert c.type == "baz"
