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

dot_node1 = """digraph {
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
    
    // Node 0
    node0 [label=<{<b>Character(0)</b> | labels: set = \\{'Character'\\}<br align="left"/>name: str = Waymar Royce<br align="left"/>}>]

    // Node 6
    node6 [label=<{<b>Allegiance(6)</b> | labels: set = \\{'Allegiance'\\}<br align="left"/>name: str = Nights Watch<br align="left"/>}>]

    // Node 453
    node453 [label=<{<b>Death(453)</b> | labels: set = \\{'Death'\\}<br align="left"/>order: int = 1<br align="left"/>}>]

    // Node 2
    node2 [label=<{<b>Character(2)</b> | labels: set = \\{'Character'\\}<br align="left"/>name: str = White Walker<br align="left"/>}>]

    // Edge 0
    node0 -> node6

    // Edge 1
    node0 -> node453

    // Edge 11
    node2 -> node0

    // Edge 4
    node2 -> node453
}"""
