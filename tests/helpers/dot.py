dot_schema1 = """digraph Schema {
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
