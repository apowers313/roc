from roc.graphdb import Node


class GotCharacter(Node, extra="forbid"):
    name: str
    # foo: int


class GotSeason(Node, extra="forbid"):
    number: int
