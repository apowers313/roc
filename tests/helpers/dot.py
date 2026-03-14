from helpers.got_ids import got_edge_id, got_node_id

dot_schema1 = """digraph {
    graph [
        fontname="Arial"
        labelloc="t"
    ]

    node [
        fontname="Arial"
        shape=record
        style=filled
        fillcolor=gray95
    ]

    edge [
        fontname="Arial"
        style=""
    ]

    // Node: Bar
    Bar [label=<{ <b>Bar</b> | ^labels: set = ['Bar']<br align="left"/>+weight: float<br align="left"/> | +print_weight(): None<br align="left"/> }>]

    // Node: Baz
    Baz [label=<{ <b>Baz</b> | ^labels: set = ['Baz']<br align="left"/> |  }>]

    // Node: Foo
    Foo [label=<{ <b>Foo</b> | ^labels: set = ['Bar', 'Foo']<br align="left"/>+name: str = Bob<br align="left"/>^weight: float<br align="left"/> | ^print_weight(): None<br align="left"/>+set_name(name: str = Uggo): str<br align="left"/> }>]
    Foo -> Bar [label="inherits" arrowhead=empty]

    // Edge: Link
    Foo -> Baz [label="Link" arrowhead=vee]
}"""


def _dot_header() -> str:
    return """digraph {
    graph [
        fontname="Arial"
        labelloc="t"
    ]

    node [
        fontname="Arial"
        shape=record
        style=filled
        fillcolor=gray95
    ]

    edge [
        fontname="Arial"
        style=""
    ]"""


def make_dot_node1() -> str:
    n0 = got_node_id(0)
    n2 = got_node_id(2)
    n6 = got_node_id(6)
    n453 = got_node_id(453)
    e0 = got_edge_id(0)
    e1 = got_edge_id(1)
    e4 = got_edge_id(4)
    e11 = got_edge_id(11)
    return f"""{_dot_header()}

    // Node {n6}
    node{n6} [label=<{{<b>Allegiance({n6})</b> | labels: set = \\{{'Allegiance'\\}}<br align="left"/>name: str = Nights Watch<br align="left"/>}}>]

    // Node {n453}
    node{n453} [label=<{{<b>Death({n453})</b> | labels: set = \\{{'Death'\\}}<br align="left"/>order: int = 1<br align="left"/>}}>]

    // Node {n2}
    node{n2} [label=<{{<b>Character({n2})</b> | labels: set = \\{{'Character'\\}}<br align="left"/>name: str = White Walker<br align="left"/>}}>]

    // Node {n0}
    node{n0} [label=<{{<b>Character({n0})</b> | labels: set = \\{{'Character'\\}}<br align="left"/>name: str = Waymar Royce<br align="left"/>}}>]

    // Edge {e0}
    node{n0} -> node{n6} [label="Edge"]

    // Edge {e1}
    node{n0} -> node{n453} [label="Edge"]

    // Edge {e11}
    node{n2} -> node{n0} [label="Edge"]

    // Edge {e4}
    node{n2} -> node{n453} [label="Edge"]
}}"""


def make_dot_node2() -> str:
    n0 = got_node_id(0)
    n2 = got_node_id(2)
    n6 = got_node_id(6)
    n453 = got_node_id(453)
    e0 = got_edge_id(0)
    e1 = got_edge_id(1)
    e4 = got_edge_id(4)
    e11 = got_edge_id(11)
    return f"""{_dot_header()}

    // Node {n6}
    node{n6} [label=<{{<b>Allegiance({n6})</b> | labels: set = \\{{'Allegiance'\\}}<br align="left"/>name: str = Nights Watch<br align="left"/>}}>]

    // Node {n453}
    node{n453} [label=<{{<b>Death({n453})</b> | labels: set = \\{{'Death'\\}}<br align="left"/>order: int = 1<br align="left"/>}}>]

    // Node {n2}
    node{n2} [label=<{{<b>Character({n2})</b> | labels: set = \\{{'Character'\\}}<br align="left"/>name: str = White Walker<br align="left"/>}}>]

    // Node {n0}
    node{n0} [label=<{{<b>Character({n0})</b> | labels: set = \\{{'Character'\\}}<br align="left"/>name: str = Waymar Royce<br align="left"/>}}> style=filled, fillcolor=red]

    // Edge {e0}
    node{n0} -> node{n6} [label="Edge"]

    // Edge {e1}
    node{n0} -> node{n453} [label="Edge"]

    // Edge {e11}
    node{n2} -> node{n0} [label="Edge"]

    // Edge {e4}
    node{n2} -> node{n453} [label="Edge"]
}}"""
