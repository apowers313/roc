from typing import Any

import atexit
import json
import os
import re
from collections.abc import Iterator

import json_fix
from gqlalchemy import Node as GQLNode
from gqlalchemy import Relationship as GQLRelationship
from icecream import ic

from roc.graphdb import GraphDB

basepath = os.path.dirname(os.path.abspath(__file__))
records_dir = os.path.abspath(os.path.join(basepath, "..", "data", "_generated"))
records_json_path = os.path.abspath(os.path.join(records_dir, "query_data.json"))


def get_records() -> dict[str, Any]:
    try:
        with open(records_json_path) as f:
            return json.load(f)  # type: ignore
    except Exception:
        return dict()


records = get_records()
records_count = 0


def normalize_whitespace(s: str) -> str:
    s = s.strip().replace("\n", " ")
    return re.sub(r"\s+", " ", s)


def save_recording() -> None:
    if records_count > 0:
        print(f"*** SAVING DATABASE RECORDING: {records_count} RECORDS ***")
    else:
        print("*** ZERO RECORDS, NOT SAVING DATABASE RECORDING ***")

    dirExists = os.path.isdir(records_dir)
    if not dirExists:
        os.makedirs(records_dir)

    with open(records_json_path, "w") as f:
        json.dump(records, f, indent=4, sort_keys=True)


def do_recording() -> None:
    GraphDB().record_callback = record_raw_query
    atexit.register(save_recording)


def patch_properties(n: Any) -> None:
    props = list(filter(lambda k: not k.startswith("_"), n.keys()))
    if len(props) > 0:
        n["_properties"] = {}
        for k in props:
            n["_properties"][k] = n[k]
            del n[k]


class QueryRecord(json.JSONEncoder):
    def __init__(self, query: str, res: Iterator[Any]):
        self.query = query
        self.res = list(res)
        # res is a list of dictonaries, where each key is a pydantic object with the 'dict()' method
        for i in range(len(self.res)):
            row = self.res[i]
            for key in row.keys():
                entry = self.res[i][key]
                if isinstance(entry, GQLNode) or isinstance(entry, GQLRelationship):
                    self.res[i][key] = self.res[i][key].dict()
                    # pydantic removes _properties, recreate it...
                    patch_properties(self.res[i][key])

    def __json__(self) -> Any:
        return {"query": self.query, "res": self.res}

    @staticmethod
    def from_dict():
        pass


def get_current_test_parts() -> list[str]:
    test_str = os.environ["PYTEST_CURRENT_TEST"]
    return test_str.split(" ")[0].split("::")


def get_current_test_record() -> dict[str, Any]:
    test_parts = get_current_test_parts()
    cur_dict = records
    for test_part in test_parts:
        if not test_part in cur_dict:
            cur_dict[test_part] = dict()
        cur_dict = cur_dict[test_part]
    return cur_dict


def add_test_record(rec: QueryRecord) -> None:
    global records_count
    records_count = records_count + 1

    cur_dict = get_current_test_record()

    i = 0
    while str(i) in cur_dict:
        i = i + 1

    cur_dict[str(i)] = rec


def record_raw_query(query: str, res: Iterator[Any]) -> None:
    query = normalize_whitespace(query)
    qr = QueryRecord(query, res)
    add_test_record(qr)


def clear_current_test_record() -> None:
    rec = get_current_test_record()
    for k in rec.keys():
        del rec[k]


prev_test = ""
curr_test_count = 0


def set_test_count() -> None:
    global prev_test, curr_test_count

    curr_test = os.environ["PYTEST_CURRENT_TEST"]
    if prev_test == curr_test:
        curr_test_count = curr_test_count + 1
    else:
        prev_test = curr_test
        curr_test_count = 0


def get_query_record(query: str) -> Iterator[Any]:
    query = normalize_whitespace(query)
    set_test_count()
    curr_test = get_current_test_record()
    curr_test = curr_test[str(curr_test_count)]
    if curr_test["query"] != query:
        exception_msg = f"""While mocking database, executed query did not match expected query: 
        EXECUTED QUERY: '{query}'
        EXPECTED QUERY: '{curr_test["query"]}'
        TEST: '{os.environ["PYTEST_CURRENT_TEST"]}'"""
        raise Exception(exception_msg)
    res = curr_test["res"]
    for row in res:
        for k in row.keys():
            if isinstance(row[k], dict):
                row[k] = type("TESTJSON", (), row[k])
    return iter(res)
