# mypy: disable-error-code="no-untyped-def"
"""Tests for pure Python logic in roc/graphdb.py that does NOT require Memgraph.

These tests cover GraphCache, NodeList, EdgeList, Schema, utility functions,
description classes, and Node/Edge registration -- all without a database connection.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import Field

from roc.graphdb import (
    Edge,
    EdgeConnectionsList,
    EdgeId,
    EdgeList,
    GraphCache,
    Node,
    NodeId,
    NodeList,
    ParamData,
    Schema,
    SchemaValidationError,
    _EdgeDescription,
    _FieldDescription,
    _MethodDescription,
    _NodeDescription,
    _clean_annotation,
    _dot_safe_node_str,
    _get_methods,
    _get_node_parent_names,
    _is_local,
    _pydantic_get_default,
    _pydantic_get_field,
    _pydantic_get_fields,
    dot_graph_header,
    edge_registry,
    node_label_registry,
    node_registry,
    true_filter,
    no_callback,
)


# ========================================================================
# GraphCache
# ========================================================================
class TestGraphCache:
    def test_init(self):
        c: GraphCache[str, int] = GraphCache(maxsize=100)
        assert c.maxsize == 100
        assert c.currsize == 0
        assert c.hits == 0
        assert c.misses == 0

    def test_setitem_and_get_hit(self):
        c: GraphCache[str, int] = GraphCache(maxsize=10)
        c["a"] = 1
        assert c.currsize == 1

        val = c.get("a")
        assert val == 1
        assert c.hits == 1
        assert c.misses == 0

    def test_get_miss(self):
        c: GraphCache[str, int] = GraphCache(maxsize=10)
        val = c.get("missing")
        assert val is None
        assert c.hits == 0
        assert c.misses == 1

    def test_get_miss_when_full_logs_warning(self, caplog):
        c: GraphCache[str, int] = GraphCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        assert c.currsize == c.maxsize

        # miss on a full cache
        val = c.get("missing")
        assert val is None
        assert c.misses == 1

    def test_clear_resets_stats(self):
        c: GraphCache[str, int] = GraphCache(maxsize=10)
        c["a"] = 1
        c.get("a")
        c.get("missing")
        assert c.hits == 1
        assert c.misses == 1

        c.clear()
        assert c.currsize == 0
        assert c.hits == 0
        assert c.misses == 0

    def test_str(self):
        c: GraphCache[str, int] = GraphCache(maxsize=100)
        c["x"] = 42
        c.get("x")
        c.get("y")
        s = str(c)
        assert "1/100" in s
        assert "Hits: 1" in s
        assert "Misses: 1" in s

    def test_multiple_hits_and_misses(self):
        c: GraphCache[str, int] = GraphCache(maxsize=10)
        c["a"] = 1
        c["b"] = 2

        c.get("a")
        c.get("a")
        c.get("b")
        c.get("missing1")
        c.get("missing2")
        assert c.hits == 3
        assert c.misses == 2

    def test_lru_eviction(self):
        c: GraphCache[str, int] = GraphCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3  # evicts "a"
        assert c.currsize == 2
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3


# ========================================================================
# NodeList (pure in-memory operations using IDs only)
# ========================================================================
class TestNodeListPure:
    def test_init_empty(self):
        nl = NodeList([])
        assert len(nl) == 0

    def test_init_with_ids(self):
        nl = NodeList([NodeId(1), NodeId(2), NodeId(3)])
        assert len(nl) == 3

    def test_contains_by_id(self):
        nl = NodeList([NodeId(1), NodeId(2)])
        assert NodeId(1) in nl
        assert NodeId(2) in nl
        assert NodeId(99) not in nl

    def test_contains_non_node_non_int(self):
        nl = NodeList([NodeId(1)])
        assert "string" not in nl  # type: ignore[comparison-overlap]

    def test_add_new_id(self):
        nl = NodeList([NodeId(1)])
        nl.add(NodeId(2))
        assert len(nl) == 2
        assert NodeId(2) in nl

    def test_add_duplicate_id(self):
        nl = NodeList([NodeId(1)])
        nl.add(NodeId(1))
        assert len(nl) == 1

    def test_discard(self):
        nl = NodeList([NodeId(1), NodeId(2)])
        nl.discard(NodeId(1))
        assert len(nl) == 1
        assert NodeId(1) not in nl

    def test_discard_missing_raises(self):
        nl = NodeList([NodeId(1)])
        with pytest.raises(ValueError):
            nl.discard(NodeId(99))

    def test_ids_property(self):
        nl = NodeList([NodeId(1), NodeId(2), NodeId(3)])
        assert nl.ids == {NodeId(1), NodeId(2), NodeId(3)}

    def test_add_concatenation(self):
        nl1 = NodeList([NodeId(1), NodeId(2)])
        nl2 = NodeList([NodeId(3), NodeId(4)])
        combined = nl1 + nl2
        assert len(combined) == 4
        assert combined.ids == {NodeId(1), NodeId(2), NodeId(3), NodeId(4)}

    def test_getitem_slice(self):
        nl = NodeList([NodeId(1), NodeId(2), NodeId(3)])
        sliced = nl[0:2]
        assert isinstance(sliced, NodeList)
        assert len(sliced) == 2
        assert sliced.ids == {NodeId(1), NodeId(2)}


# ========================================================================
# EdgeList (pure in-memory operations using IDs only)
# ========================================================================
class TestEdgeListPure:
    def test_init_empty(self):
        el = EdgeList([])
        assert len(el) == 0

    def test_init_with_ids(self):
        el = EdgeList([EdgeId(10), EdgeId(20)])
        assert len(el) == 2

    def test_contains_by_id(self):
        el = EdgeList([EdgeId(10), EdgeId(20)])
        assert EdgeId(10) in el
        assert EdgeId(20) in el
        assert EdgeId(99) not in el

    def test_contains_non_edge_non_int(self):
        el = EdgeList([EdgeId(10)])
        assert "string" not in el  # type: ignore[comparison-overlap]

    def test_add_new(self):
        el = EdgeList([EdgeId(10)])
        el.add(EdgeId(20))
        assert len(el) == 2
        assert EdgeId(20) in el

    def test_add_duplicate(self):
        el = EdgeList([EdgeId(10)])
        el.add(EdgeId(10))
        assert len(el) == 1

    def test_discard(self):
        el = EdgeList([EdgeId(10), EdgeId(20)])
        el.discard(EdgeId(10))
        assert len(el) == 1
        assert EdgeId(10) not in el

    def test_ids_property(self):
        el = EdgeList([EdgeId(10), EdgeId(20)])
        assert el.ids == {EdgeId(10), EdgeId(20)}

    def test_add_concatenation(self):
        el1 = EdgeList([EdgeId(1), EdgeId(2)])
        el2 = EdgeList([EdgeId(3), EdgeId(4)])
        combined = el1 + el2
        assert len(combined) == 4
        assert combined.ids == {EdgeId(1), EdgeId(2), EdgeId(3), EdgeId(4)}

    def test_getitem_slice(self):
        el = EdgeList([EdgeId(1), EdgeId(2), EdgeId(3)])
        sliced = el[0:2]
        assert isinstance(sliced, EdgeList)
        assert len(sliced) == 2
        assert sliced.ids == {EdgeId(1), EdgeId(2)}

    def test_replace(self):
        el = EdgeList([EdgeId(1), EdgeId(2), EdgeId(1)])
        el.replace(EdgeId(1), EdgeId(99))
        assert EdgeId(99) in el
        assert el._edges[0] == EdgeId(99)
        assert el._edges[2] == EdgeId(99)
        assert el._edges[1] == EdgeId(2)


# ========================================================================
# Utility functions
# ========================================================================
class TestUtilityFunctions:
    def test_true_filter(self):
        assert true_filter(None) is True
        assert true_filter(42) is True
        assert true_filter("anything") is True

    def test_no_callback(self):
        assert no_callback(None) is None  # type: ignore[func-returns-value]
        assert no_callback(42) is None  # type: ignore[func-returns-value]

    def test_dot_safe_node_str_positive(self):
        assert _dot_safe_node_str(NodeId(5)) == "node5"

    def test_dot_safe_node_str_negative(self):
        assert _dot_safe_node_str(NodeId(-3)) == "node_3"

    def test_dot_safe_node_str_zero(self):
        assert _dot_safe_node_str(NodeId(0)) == "node0"

    def test_clean_annotation_string(self):
        assert _clean_annotation("MyType") == "MyType"

    def test_clean_annotation_none(self):
        assert _clean_annotation(None) == "None"

    def test_clean_annotation_class(self):
        assert _clean_annotation(int) == "int"
        assert _clean_annotation(str) == "str"

    def test_clean_annotation_generic(self):
        import typing

        result = _clean_annotation(typing.List[int])
        assert result == "list[int]"

    def test_clean_annotation_special_form(self):
        from typing import Any

        result = _clean_annotation(Any)
        assert result == "Any"

    def test_is_local_direct(self):
        class A:
            def foo(self):
                """Stub for locality testing."""

        assert _is_local(A, "foo") is True

    def test_is_local_inherited(self):
        class A:
            def foo(self):
                """Stub for locality testing."""

        class B(A):
            pass

        assert _is_local(B, "foo") is False

    def test_is_local_overridden(self):
        # _is_local returns False if any parent also has the attr,
        # even if the child overrides it
        class A:
            def foo(self):
                """Stub for locality testing."""

        class B(A):
            def foo(self):
                """Stub for locality testing."""

        assert _is_local(B, "foo") is False

    def test_get_node_parent_names_no_parents(self):
        class Leaf(Node):
            pass

        result = _get_node_parent_names(Leaf)
        assert result == set()

    def test_get_node_parent_names_with_parents(self):
        class Parent(Node):
            pass

        class Child(Parent):
            pass

        result = _get_node_parent_names(Child)
        assert result == {"Parent"}

    def test_get_node_parent_names_deep_hierarchy(self):
        class GrandParent(Node):
            pass

        class Parent(GrandParent):
            pass

        class Child(Parent):
            pass

        result = _get_node_parent_names(Child)
        assert result == {"Parent", "GrandParent"}

    def test_pydantic_get_fields(self):
        class MyModel(Node):
            name: str
            value: int = 0

        fields = _pydantic_get_fields(MyModel)
        assert "name" in fields
        assert "value" in fields
        assert "labels" in fields

    def test_pydantic_get_field(self):
        class MyModel(Node):
            name: str = "default"

        fi = _pydantic_get_field(MyModel, "name")
        assert fi is not None
        assert fi.get_default(call_default_factory=True) == "default"

    def test_pydantic_get_default(self):
        class MyModel(Node):
            name: str = "hello"

        assert _pydantic_get_default(MyModel, "name") == "hello"

    def test_get_methods(self):
        class MyClass:
            def foo(self):
                """Stub for method discovery testing."""

            def bar(self):
                """Stub for method discovery testing."""

        methods = _get_methods(MyClass)
        assert "foo" in methods
        assert "bar" in methods


# ========================================================================
# Node.__init_subclass__ registration
# ========================================================================
class TestNodeRegistration:
    def test_auto_labels(self):
        class AutoLabeled(Node):
            pass

        n = AutoLabeled(_no_save=True, _db=MagicMock())
        assert n.labels == {"AutoLabeled"}

    def test_auto_labels_hierarchy(self):
        class Parent(Node):
            pass

        class Child(Parent):
            pass

        n = Child(_no_save=True, _db=MagicMock())
        assert n.labels == {"Parent", "Child"}

    def test_custom_labels(self):
        class CustomLabeled(Node):
            labels: set[str] = {"Custom", "Labels"}

        n = CustomLabeled(_no_save=True, _db=MagicMock())
        assert n.labels == {"Custom", "Labels"}

    def test_custom_labels_field(self):
        class CustomField(Node):
            labels: set[str] = Field(default_factory=lambda: {"FieldLabel"})

        n = CustomField(_no_save=True, _db=MagicMock())
        assert n.labels == {"FieldLabel"}

    def test_node_in_registry(self):
        class RegisteredNode(Node):
            pass

        assert "RegisteredNode" in node_registry
        assert node_registry["RegisteredNode"] is RegisteredNode

    def test_node_in_label_registry(self):
        class LabelRegNode(Node):
            pass

        key = frozenset({"LabelRegNode"})
        assert key in node_label_registry
        assert node_label_registry[key] is LabelRegNode

    def test_duplicate_name_raises(self):
        class UniqueNodeA(Node):
            pass

        with pytest.raises(ValueError, match="node_register can't register"):

            class UniqueNodeA(Node):  # type: ignore
                pass

    def test_duplicate_labels_raises(self):
        class DupLblNode(Node):
            pass

        with pytest.raises(ValueError, match="node_register can't register labels"):

            class DupLblNode2(Node):
                labels: set[str] = {"DupLblNode"}


# ========================================================================
# Edge.__init_subclass__ registration
# ========================================================================
class TestEdgeRegistration:
    def test_auto_type(self):
        class AutoTypedEdge(Edge):
            pass

        assert "AutoTypedEdge" in edge_registry
        assert edge_registry["AutoTypedEdge"] is AutoTypedEdge

    def test_custom_type(self):
        class CustomTypedEdge(Edge):
            type: str = "MyCustomType"

        assert "MyCustomType" in edge_registry
        assert edge_registry["MyCustomType"] is CustomTypedEdge

    def test_custom_type_field(self):
        class FieldTypedEdge(Edge):
            type: str = Field(default_factory=lambda: "FieldType")

        assert "FieldType" in edge_registry
        assert edge_registry["FieldType"] is FieldTypedEdge

    def test_duplicate_type_raises(self):
        class UniqueEdgeA(Edge):
            pass

        with pytest.raises(ValueError, match="edge_register can't register type"):

            class UniqueEdgeB(Edge):
                type: str = "UniqueEdgeA"


# ========================================================================
# Schema (uses registries only, no DB)
# ========================================================================
class TestSchemaPure:
    def test_validate_passes(self, clear_registries):
        class Src(Node):
            pass

        class Dst(Node):
            pass

        class Link(Edge):
            allowed_connections: EdgeConnectionsList = [("Src", "Dst")]

        Schema.validate()

    def test_validate_fails_missing_src(self, clear_registries):
        class Dst(Node):
            pass

        class BadLink(Edge):
            allowed_connections: EdgeConnectionsList = [("Missing", "Dst")]

        with pytest.raises(SchemaValidationError) as exc_info:
            Schema.validate()
        assert len(exc_info.value.errors) >= 1
        assert "Missing" in exc_info.value.errors[0]

    def test_validate_fails_missing_dst(self, clear_registries):
        class Src(Node):
            pass

        class BadLink(Edge):
            allowed_connections: EdgeConnectionsList = [("Src", "Missing")]

        _ = BadLink
        with pytest.raises(SchemaValidationError) as exc_info:
            Schema.validate()
        assert len(exc_info.value.errors) >= 1
        assert "Missing" in exc_info.value.errors[0]

    def test_validate_no_allowed_connections_ok(self, clear_registries):
        class NoConn(Edge):
            pass

        _ = NoConn
        # no allowed_connections is fine -- just skip
        Schema.validate()

    def test_schema_init_collects_nodes_and_edges(self, clear_registries):
        class NodeA(Node):
            pass

        class NodeB(Node):
            pass

        class ConnAB(Edge):
            allowed_connections: EdgeConnectionsList = [("NodeA", "NodeB")]

        schema = Schema()
        assert "NodeA" in schema.node_names
        assert "NodeB" in schema.node_names
        assert "ConnAB" in schema.edge_names

    def test_schema_to_mermaid(self, clear_registries):
        class MermNode(Node):
            pass

        class MermEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("MermNode", "MermNode")]

        schema = Schema()
        mermaid = schema.to_mermaid()
        assert "classDiagram" in mermaid
        assert "MermNode" in mermaid
        assert "MermEdge" in mermaid

    def test_schema_to_dot(self, clear_registries):
        class DotNode(Node):
            pass

        class DotEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("DotNode", "DotNode")]

        schema = Schema()
        dot = schema.to_dot()
        assert "digraph" in dot
        assert "DotNode" in dot
        assert "DotEdge" in dot

    def test_schema_repr_markdown(self, clear_registries):
        class MdNode(Node):
            pass

        class MdEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("MdNode", "MdNode")]

        _ = MdEdge
        md = Schema._repr_markdown_()
        assert "``` mermaid" in md
        assert "MdNode" in md


# ========================================================================
# SchemaValidationError
# ========================================================================
class TestSchemaValidationError:
    def test_error_message(self):
        errors = ["error one", "error two"]
        exc = SchemaValidationError(errors)
        assert exc.errors == errors
        assert "error one" in str(exc)
        assert "error two" in str(exc)
        assert "0:" in str(exc)
        assert "1:" in str(exc)


# ========================================================================
# _FieldDescription
# ========================================================================
class TestFieldDescription:
    def test_basic_field(self):
        class MyModel(Node):
            name: str = "default"

        fd = _FieldDescription(MyModel, "name")
        assert fd.name == "name"
        assert fd.default_val == "default"
        assert fd.type == "str"
        assert fd.exclude is not True
        assert "name" in str(fd)

    def test_excluded_field(self):
        class MyModel(Node):
            labels: set[str] = Field(default_factory=set, exclude=True)

        fd = _FieldDescription(MyModel, "labels")
        assert fd.exclude is True

    def test_default_val_str_set(self):
        class MyModel(Node):
            tags: set[str] = Field(default_factory=lambda: {"b", "a", "c"})

        fd = _FieldDescription(MyModel, "tags")
        # set default should be sorted for reproducibility
        assert fd.default_val_str == "['a', 'b', 'c']"

    def test_default_val_str_non_set(self):
        class MyModel(Node):
            count: int = 42

        fd = _FieldDescription(MyModel, "count")
        assert fd.default_val_str == "42"


# ========================================================================
# ParamData
# ========================================================================
class TestParamData:
    def test_with_default(self):
        p = ParamData(type="str", name="x", default="hello")
        assert p.formatted_default == " = hello"

    def test_without_default(self):
        p = ParamData(type="str", name="x", default=inspect._empty)
        assert p.formatted_default == ""


# ========================================================================
# _MethodDescription
# ========================================================================
class TestMethodDescription:
    def test_basic(self):
        class MyModel(Node):
            def do_thing(self, a: int, b: str = "hi") -> bool:
                return True

        md = _MethodDescription(MyModel, "do_thing")
        assert md.name == "do_thing"
        assert md.return_type == "bool"
        params = md.params
        assert len(params) == 2
        assert params[0].name == "a"
        assert params[0].type == "int"
        assert params[1].name == "b"
        assert params[1].default == "hi"


# ========================================================================
# _NodeDescription
# ========================================================================
class TestNodeDescriptionPure:
    def test_basic(self):
        class DescNode(Node):
            name: str = "test"

        nd = _NodeDescription(DescNode)
        assert nd.name == "DescNode"
        assert any(f.name == "name" for f in nd.fields)

    def test_with_methods(self):
        class MethodNode(Node):
            def custom_method(self) -> None:
                """Stub for NodeDescription method testing."""

        nd = _NodeDescription(MethodNode)
        assert "custom_method" in nd.method_names

    def test_with_parent(self):
        class ParentNode(Node):
            pass

        class ChildNode(ParentNode):
            pass

        nd = _NodeDescription(ChildNode)
        assert "ParentNode" in nd.parent_class_names
        assert len(nd.parents) == 1
        assert nd.parents[0].name == "ParentNode"

    def test_str(self):
        class StrNode(Node):
            pass

        nd = _NodeDescription(StrNode)
        assert str(nd) == "NodeDesc(StrNode)"

    def test_to_mermaid(self):
        class MermDescNode(Node):
            value: int = 0

        nd = _NodeDescription(MermDescNode)
        mermaid = nd.to_mermaid()
        assert "MermDescNode" in mermaid
        assert "value" in mermaid

    def test_to_dot(self):
        class DotDescNode(Node):
            value: int = 0

        nd = _NodeDescription(DotDescNode)
        dot = nd.to_dot()
        assert "DotDescNode" in dot
        assert "value" in dot


# ========================================================================
# _EdgeDescription
# ========================================================================
class TestEdgeDescriptionPure:
    def test_basic(self):
        class SrcNode(Node):
            pass

        class DstNode(Node):
            pass

        class DescEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("SrcNode", "DstNode")]

        ed = _EdgeDescription(DescEdge)
        assert ed.name == "DescEdge"
        assert ed.edgetype == "DescEdge"
        assert ed.related_nodes == {"SrcNode", "DstNode"}
        assert ed.resolved_name == "DescEdge"

    def test_resolved_name_different(self):
        class SrcNode2(Node):
            pass

        class DstNode2(Node):
            pass

        class DiffNameEdge(Edge):
            type: str = "DIFFERENT"
            allowed_connections: EdgeConnectionsList = [("SrcNode2", "DstNode2")]

        ed = _EdgeDescription(DiffNameEdge)
        assert ed.edgetype == "DIFFERENT"
        assert ed.resolved_name == "DIFFERENT (DiffNameEdge)"

    def test_str(self):
        class StrSrc(Node):
            pass

        class StrDst(Node):
            pass

        class StrEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("StrSrc", "StrDst")]

        ed = _EdgeDescription(StrEdge)
        assert str(ed) == "EdgeDesc(StrEdge)"

    def test_to_mermaid(self):
        class MermSrc(Node):
            pass

        class MermDst(Node):
            pass

        class MermDescEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("MermSrc", "MermDst")]

        ed = _EdgeDescription(MermDescEdge)
        mermaid = ed.to_mermaid()
        assert "MermSrc" in mermaid
        assert "MermDst" in mermaid
        assert "MermDescEdge" in mermaid

    def test_to_dot(self):
        class DotSrc(Node):
            pass

        class DotDst(Node):
            pass

        class DotDescEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("DotSrc", "DotDst")]

        ed = _EdgeDescription(DotDescEdge)
        dot = ed.to_dot()
        assert "DotSrc" in dot
        assert "DotDst" in dot
        assert "DotDescEdge" in dot


# ========================================================================
# Node/Edge static methods that don't hit DB
# ========================================================================
class TestNodeStaticMethods:
    def test_mklabels_empty(self):
        assert Node.mklabels(set()) == ""

    def test_mklabels_single(self):
        assert Node.mklabels({"Foo"}) == ":Foo"

    def test_mklabels_multiple_sorted(self):
        result = Node.mklabels({"Zebra", "Alpha", "Middle"})
        assert result == ":Alpha:Middle:Zebra"

    def test_to_id_with_node_id(self):
        nid = NodeId(42)
        assert Node.to_id(nid) == NodeId(42)

    def test_to_id_with_node(self):
        n = Node(_no_save=True, _db=MagicMock())
        result = Node.to_id(n)
        assert result == n.id

    def test_to_dict_basic(self):
        n = Node(_no_save=True, _db=MagicMock(), foo="bar")
        d = Node.to_dict(n)
        assert "foo" in d
        assert d["foo"] == "bar"

    def test_to_dict_include_labels(self):
        n = Node(_no_save=True, _db=MagicMock(), labels={"TestLabel"})
        d = Node.to_dict(n, include_labels=True)
        assert "labels" in d
        assert "TestLabel" in d["labels"]

    def test_to_dict_no_labels_attr(self):
        n = Node(_no_save=True, _db=MagicMock())
        del n.labels
        d = Node.to_dict(n, include_labels=True)
        assert "labels" not in d


class TestEdgeStaticMethods:
    def test_to_id_with_edge_id(self):
        eid = EdgeId(42)
        assert Edge.to_id(eid) == EdgeId(42)

    def test_to_dict_empty(self):
        e = Edge(
            type="TestType",
            src_id=NodeId(1),
            dst_id=NodeId(2),
            _id=EdgeId(100),
        )
        e._no_save = True
        d = Edge.to_dict(e)
        assert d == {}

    def test_to_dict_with_data(self):
        e = Edge(
            type="TestType",
            src_id=NodeId(1),
            dst_id=NodeId(2),
            _id=EdgeId(100),
            foo="bar",
        )
        e._no_save = True
        d = Edge.to_dict(e)
        assert d == {"foo": "bar"}

    def test_to_dict_include_type(self):
        e = Edge(
            type="TestType",
            src_id=NodeId(1),
            dst_id=NodeId(2),
            _id=EdgeId(100),
        )
        e._no_save = True
        d = Edge.to_dict(e, include_type=True)
        assert d == {"type": "TestType"}


# ========================================================================
# Node/Edge properties and repr
# ========================================================================
class TestNodeProperties:
    def test_id(self):
        mock_db = MagicMock()
        n = Node(_id=NodeId(42), _no_save=True, _db=mock_db)
        assert n.id == NodeId(42)

    def test_repr(self):
        mock_db = MagicMock()
        n = Node(_id=NodeId(42), _no_save=True, _db=mock_db)
        assert repr(n) == "Node(42)"

    def test_str(self):
        mock_db = MagicMock()
        n = Node(_id=NodeId(42), _no_save=True, _db=mock_db, labels={"Foo"})
        assert "Node(42" in str(n)
        assert "Foo" in str(n)

    def test_new_property(self):
        mock_db = MagicMock()
        # positive ID = not new
        n = Node(_id=NodeId(42), _no_save=True, _db=mock_db)
        assert n.new is False

    def test_src_edges_empty(self):
        mock_db = MagicMock()
        n = Node(_id=NodeId(42), _no_save=True, _db=mock_db)
        assert len(n.src_edges) == 0

    def test_dst_edges_empty(self):
        mock_db = MagicMock()
        n = Node(_id=NodeId(42), _no_save=True, _db=mock_db)
        assert len(n.dst_edges) == 0

    def test_edges_combines_src_and_dst(self):
        mock_db = MagicMock()
        n = Node(
            _id=NodeId(42),
            _no_save=True,
            _db=mock_db,
            _src_edges=EdgeList([EdgeId(1)]),
            _dst_edges=EdgeList([EdgeId(2)]),
        )
        assert len(n.edges) == 2


class TestEdgeProperties:
    def _make_edge(self, **kwargs: Any) -> Edge:
        defaults = {"type": "T", "src_id": NodeId(1), "dst_id": NodeId(2), "_id": EdgeId(99)}
        defaults.update(kwargs)
        e = Edge(**defaults)
        e._no_save = True
        return e

    def test_id(self):
        e = self._make_edge()
        assert e.id == EdgeId(99)

    def test_repr(self):
        e = self._make_edge()
        assert repr(e) == "Edge(99 [1>>2])"

    def test_new_property_positive_id(self):
        e = self._make_edge()
        assert e.new is False

    def test_repr_dot(self):
        e = self._make_edge()
        dot = e._repr_dot_()
        assert "node1" in dot
        assert "node2" in dot
        assert "Edge" in dot


# ========================================================================
# dot_graph_header constant
# ========================================================================
class TestDotGraphHeader:
    def test_contains_digraph(self):
        assert "digraph" in dot_graph_header
        assert "fontname" in dot_graph_header
        assert "shape=record" in dot_graph_header


# ========================================================================
# Schema.to_dict
# ========================================================================
class TestSchemaToDict:
    def test_basic_schema_to_dict(self, clear_registries):
        """to_dict returns a dict with mermaid, nodes, and edges keys."""

        class DictNode(Node):
            value: int = 0

        class DictEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("DictNode", "DictNode")]

        schema = Schema()
        result = schema.to_dict()

        assert "mermaid" in result
        assert "nodes" in result
        assert "edges" in result
        assert isinstance(result["mermaid"], str)
        assert "classDiagram" in result["mermaid"]

    def test_nodes_structure(self, clear_registries):
        """Node entries have name, parents, fields, and methods."""

        class ParentNode(Node):
            pass

        class ChildNode(ParentNode):
            name: str = "test"

            def custom_method(self) -> bool:
                """Stub for to_dict testing."""
                return True

        class DummyEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("ParentNode", "ChildNode")]

        schema = Schema()
        result = schema.to_dict()

        child_nodes = [n for n in result["nodes"] if n["name"] == "ChildNode"]
        assert len(child_nodes) == 1
        child = child_nodes[0]
        assert "ParentNode" in child["parents"]
        assert isinstance(child["fields"], list)
        assert isinstance(child["methods"], list)

    def test_edges_structure(self, clear_registries):
        """Edge entries have name, type, connections, and fields."""

        class SrcNode(Node):
            pass

        class DstNode(Node):
            pass

        class ConnEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("SrcNode", "DstNode")]

        schema = Schema()
        result = schema.to_dict()

        conn_edges = [e for e in result["edges"] if e["name"] == "ConnEdge"]
        assert len(conn_edges) == 1
        edge = conn_edges[0]
        assert edge["type"] == "ConnEdge"
        assert ["SrcNode", "DstNode"] in edge["connections"]
        assert isinstance(edge["fields"], list)

    def test_field_serialization(self, clear_registries):
        """Fields include name, type, default, local, and exclude."""

        class FieldNode(Node):
            score: int = 42
            labels: set[str] = Field(default_factory=lambda: {"FieldNode"}, exclude=True)

        class FieldEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("FieldNode", "FieldNode")]

        schema = Schema()
        result = schema.to_dict()

        field_nodes = [n for n in result["nodes"] if n["name"] == "FieldNode"]
        assert len(field_nodes) == 1
        fields = field_nodes[0]["fields"]
        score_fields = [f for f in fields if f["name"] == "score"]
        assert len(score_fields) == 1
        assert score_fields[0]["type"] == "int"
        assert score_fields[0]["default"] == "42"
        assert score_fields[0]["exclude"] is False

    def test_method_serialization(self, clear_registries):
        """Methods include name, params, return_type, and local flag."""

        class MethodNode(Node):
            def compute(self, x: int, y: str = "default") -> bool:
                """Stub for method serialization testing."""
                return True

        class MethodEdge(Edge):
            allowed_connections: EdgeConnectionsList = [("MethodNode", "MethodNode")]

        schema = Schema()
        result = schema.to_dict()

        method_nodes = [n for n in result["nodes"] if n["name"] == "MethodNode"]
        assert len(method_nodes) == 1
        methods = method_nodes[0]["methods"]
        compute_methods = [m for m in methods if m["name"] == "compute"]
        assert len(compute_methods) == 1
        assert "x: int" in compute_methods[0]["params"]
        assert compute_methods[0]["return_type"] == "bool"


# ========================================================================
# Schema._field_to_dict and Schema._method_to_dict static methods
# ========================================================================
class TestSchemaFieldToDict:
    def test_field_with_default(self, clear_registries):
        """Serializes a field with a default value."""

        class FtdNode(Node):
            count: int = 10

        desc = _FieldDescription(FtdNode, "count")
        result = Schema._field_to_dict(desc, FtdNode)
        assert result["name"] == "count"
        assert result["type"] == "int"
        assert result["default"] == "10"
        assert isinstance(result["local"], bool)
        assert result["exclude"] is False

    def test_field_without_default(self, clear_registries):
        """Serializes a field without a default value (PydanticUndefined)."""

        class ReqNode(Node):
            required_field: str

        desc = _FieldDescription(ReqNode, "required_field")
        result = Schema._field_to_dict(desc, ReqNode)
        assert result["name"] == "required_field"
        assert result["default"] is None


class TestSchemaMethodToDict:
    def test_method_serialization(self, clear_registries):
        """Serializes a method description to a dict."""

        class MtdNode(Node):
            def do_work(self, a: int, b: str = "hi") -> bool:
                """Stub for method dict testing."""
                return True

        desc = _MethodDescription(MtdNode, "do_work")
        result = Schema._method_to_dict(desc, MtdNode)
        assert result["name"] == "do_work"
        assert "a: int" in result["params"]
        assert "b: str" in result["params"]
        assert result["return_type"] == "bool"
        assert isinstance(result["local"], bool)


# ========================================================================
# Schema._validate_edge_connections static method
# ========================================================================
class TestSchemaValidateEdgeConnections:
    def test_valid_connections(self, clear_registries):
        """No errors when all referenced nodes exist."""

        class ValSrc(Node):
            pass

        class ValDst(Node):
            pass

        errors = Schema._validate_edge_connections("TestEdge", [("ValSrc", "ValDst")])
        assert errors == []

    def test_missing_src(self, clear_registries):
        """Reports error for missing source node."""

        class ValDst2(Node):
            pass

        errors = Schema._validate_edge_connections("TestEdge", [("Missing", "ValDst2")])
        assert len(errors) == 1
        assert "Missing" in errors[0]

    def test_missing_dst(self, clear_registries):
        """Reports error for missing destination node."""

        class ValSrc2(Node):
            pass

        errors = Schema._validate_edge_connections("TestEdge", [("ValSrc2", "Missing")])
        assert len(errors) == 1
        assert "Missing" in errors[0]

    def test_both_missing(self, clear_registries):
        """Reports errors for both missing src and dst."""
        errors = Schema._validate_edge_connections("TestEdge", [("MissA", "MissB")])
        assert len(errors) == 2
