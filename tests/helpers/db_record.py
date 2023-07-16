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
records_dir = os.path.abspath(os.path.join(basepath, "..", "helpers", "_generated"))
records_json_path = os.path.abspath(os.path.join(records_dir, "query_data.json"))


def get_records() -> dict[str, Any]:
    try:
        with open(records_json_path) as f:
            return json.load(f)  # type: ignore
    except FileNotFoundError:
        return dict()


# records = get_records()
records_count = 0
records: dict[str, Any] = {}


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

    def __json__(self) -> Any:
        return {"query": self.query, "res": self.res}

    @staticmethod
    def from_dict():
        pass


def add_test_record(rec: QueryRecord) -> None:
    global records_count
    records_count = records_count + 1

    test_str = os.environ["PYTEST_CURRENT_TEST"]
    test_path = test_str.split(" ")[0].split("::")
    cur_dict = records
    for test_part in test_path:
        if not test_part in cur_dict:
            cur_dict[test_part] = dict()
        cur_dict = cur_dict[test_part]

    i = 0
    while str(i) in cur_dict:
        i = i + 1

    cur_dict[str(i)] = rec


def record_raw_query(query: str, res: Iterator[Any]) -> None:
    query = normalize_whitespace(query)
    qr = QueryRecord(query, res)
    add_test_record(qr)


# def get_query_record(query: str) -> Iterator[Any]:
#     pass
