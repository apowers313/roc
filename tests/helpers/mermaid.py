mermaid_schema1 = """classDiagram

    %% Node: Bar
    Bar: ^set labels = ['Bar']
    Bar: +float weight
    Bar: +print_weight() None

    %% Node: Baz
    Baz: ^set labels = ['Baz']

    %% Node: Foo
    Foo: ^set labels = ['Bar', 'Foo']
    Foo: +str name = Bob
    Foo: ^float weight
    Foo: ^print_weight() None
    Foo: +set_name(str name = Uggo) str
    Foo ..|> Bar: inherits

    %% Edge: Link
    Foo --> Baz: Link
"""
