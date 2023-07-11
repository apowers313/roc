import json

import pytest

from roc.graphdb import Edge, GraphDB, Node


def load_json(d: dict[str, object], k: str, file: str) -> None:
    with open("tests/data/" + file) as f:
        j = json.load(f)
        o = type("JSON", (), j)
        d[k] = o


got_data: dict[str, object] = dict()

load_json(got_data, "edge0", "db/got/edge_0.json")
load_json(got_data, "edge1", "db/got/edge_1.json")
load_json(got_data, "edge11", "db/got/edge_11.json")
load_json(got_data, "node0", "db/got/node_0.json")


@pytest.fixture
def mock_node0(mocker):
    ret = iter(
        [
            [{"n": got_data["node0"], "e": got_data["edge1"]}],
            {{"n": got_data["node0"], "e": got_data["edge0"]}},
            {{"n": got_data["node0"], "e": got_data["edge11"]}},
        ]
    )
    mocker.patch.object(GraphDB, "raw_query", return_value=ret)


@pytest.fixture
def clear_cache():
    Node.get_cache_control().clear()
    Edge.get_cache_control().clear()
