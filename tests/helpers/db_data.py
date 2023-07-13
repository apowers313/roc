from typing import Any, Callable, Type

import json
import re
from collections.abc import Iterator


def normalize_whitespace(s: str) -> str:
    s = s.strip().replace("\n", " ")
    return re.sub(r"\s+", " ", s)


# def load_json(d: dict[str, object], k: str, file: str) -> None:
#     with open("tests/data/" + file) as f:
#         j = json.load(f)
#         o: type[Any] = type("TESTJSON", (), j)
#         # a list of labels is actually a Set, but JSON doesn't do sets so do conversion
#         if hasattr(o, "_labels") and isinstance(o._labels, list):
#             o._labels = set(o._labels)

#         d[k] = o


def get_json(file: str) -> type[Any]:
    with open("tests/data/db/got/" + file + ".json") as f:
        j = json.load(f)
        o: type[Any] = type("TESTJSON", (), j)
        # a list of labels is actually a Set, but JSON doesn't do sets so do conversion
        if hasattr(o, "_labels") and isinstance(o._labels, list):
            o._labels = set(o._labels)

        return o


# got_data: dict[str, Any] = dict()

# load_json(got_data, "edge0", "db/got/edge_0.json")
# load_json(got_data, "edge1", "db/got/edge_1.json")
# load_json(got_data, "edge11", "db/got/edge_11.json")
# load_json(got_data, "node0", "db/got/node_0.json")


def partial_edge(edge_name: str) -> dict[str, Any]:
    e = get_json(edge_name)
    return {
        "e_id": e._id,
        "e_start": e._start_node_id,
        "e_end": e._end_node_id,
    }


db_query_mapping: dict[str, Callable[[], Iterator[Any]]] = {}


def add_db_query(query: str, res: Callable[[], Iterator[Any]]) -> None:
    db_query_mapping[normalize_whitespace(query)] = res


###
# Queries
###

node0_query = normalize_whitespace(
    """
                MATCH (n)-[e]-(m) WHERE id(n) = 0
                RETURN n, e, id(e) as e_id, id(startNode(e)) as e_start, id(endNode(e)) as e_end
                """
)

node0_res = lambda: iter(
    [
        {"n": get_json("node_0"), **partial_edge("edge_0")},
        {"n": get_json("node_0"), **partial_edge("edge_1")},
        {"n": get_json("node_0"), **partial_edge("edge_11")},
    ]
)

edge0_query = "MATCH (n)-[e]-(m) WHERE id(e) = 0 RETURN e LIMIT 1"
edge1_query = "MATCH (n)-[e]-(m) WHERE id(e) = 1 RETURN e LIMIT 1"
edge11_query = "MATCH (n)-[e]-(m) WHERE id(e) = 11 RETURN e LIMIT 1"

edge0_res = lambda: iter([{"e": get_json("edge_0")}])
edge1_res = lambda: iter([{"e": get_json("edge_1")}])
edge11_res = lambda: iter([{"e": get_json("edge_11")}])


add_db_query(node0_query, node0_res)
add_db_query(edge0_query, edge0_res)
add_db_query(edge1_query, edge1_res)
add_db_query(edge11_query, edge11_res)
