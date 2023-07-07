import pytest

from roc.graphdb import GraphDB


@pytest.mark.skip(reason="skip until mocks are added")
def test_graphdb_connect():
    db = GraphDB()
    db.connect()
    res = db.query(
        """
        MATCH path=(:Country { iso_2_code: "DE" })<-[:RELATED_TO]-()-[]->()
        RETURN path;
        """
    )
    print(res)
    print(repr(res))
    assert res != None
    for row in res:
        print(repr(row))
