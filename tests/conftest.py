import pytest
from helpers.db_record import do_recording

from roc.event import EventBus
from roc.graphdb import Edge, GraphDB, Node

LIVE_DB = True
RECORD_DB = False

if RECORD_DB:
    do_recording()


# def mock_raw_fetch(db: Any, query: str, *, params: dict[str, Any] | None = None) -> Iterator[Any]:
#     return get_query_record(query)


# def mock_raw_execute(db: Any, query: str, *, params: dict[str, Any] | None = None) -> None:
#     pass


@pytest.fixture(autouse=True)
def clear_cache():
    yield

    node_cache = Node.cache_control.cache
    edge_cache = Edge.cache_control.cache
    for n in node_cache:
        node_cache[n].no_save = True
    for e in edge_cache:
        edge_cache[e].no_save = True

    Node.cache_control.clear()
    Edge.cache_control.clear()


@pytest.fixture
def mock_db(clear_cache):
    pass
    # if not LIVE_DB:
    #     db = GraphDB()
    #     db.db = None
    #     with mock.patch.object(GraphDB, "raw_fetch", new=mock_raw_fetch):
    #         with mock.patch.object(GraphDB, "raw_execute", new=mock_raw_execute):
    #             yield
    # else:
    #     if RECORD_DB:
    #         clear_current_test_record()
    #     yield


@pytest.fixture
def eb_reset():
    EventBus.clear_names()


@pytest.fixture(scope="session", autouse=True)
def clear_db():
    yield
    if LIVE_DB:
        db = GraphDB()
        # delete all test nodes (which may have edges that need to be detached)
        db.raw_execute("MATCH (n:TestNode) DETACH DELETE n")
        # delete all nodes without relationships
        db.raw_execute("MATCH (n) WHERE degree(n) = 0 DELETE n")


# node_del_mock: Any = None
# edge_del_mock: Any = None


# @pytest.fixture(scope="session", autouse=True)
# def no_del():
#     global node_del_mock
#     global edge_del_mock
#     node_del_mock = mock.patch.object(Node, "__del__")
#     edge_del_mock = mock.patch.object(Edge, "__del__")
#     node_del_mock.start()
#     edge_del_mock.start()
#     yield
#     node_del_mock.stop()
#     edge_del_mock.stop()


# @pytest.fixture
# def allow_del():
#     if node_del_mock:
#         print("--- STOPPING NODE DEL MOCK")
#         node_del_mock.stop()

#     if edge_del_mock:
#         print("--- STOPPING EDGE DEL MOCK")
#         edge_del_mock.stop()

#     yield

#     if node_del_mock:
#         print("+++ STARTING NODE DEL MOCK")
#         node_del_mock.start()

#     if edge_del_mock:
#         print("+++ STARTING EDGE DEL MOCK")
#         edge_del_mock.start()
