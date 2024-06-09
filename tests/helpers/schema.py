from roc.graphdb import Node


class GotCharacter(Node, extra="ignore"):
    name: str


class GotSeason(Node, extra="forbid"):
    number: int
