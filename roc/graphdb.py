from typing import Any, Dict

from collections.abc import Iterator

from gqlalchemy import Memgraph

from roc.config import settings


class GraphDB:
    def __init__(self):
        self.host = settings.db_host
        self.port = settings.db_port
        self.db = None

    def connect(self):
        """Connects to the database. The host and port for the database are specified through the config variables 'db_host' and 'db_port' (respectively).

        Example:
            >>> db = GraphDB()
            >>> db.connect()
        """
        self.db = Memgraph(host=self.host, port=self.port)

    def query(self, query: str, *, fetch: bool = True) -> Iterator[dict[str, Any]] | None:
        if not self.db:
            raise Exception("database not connected")

        if fetch:
            return self.db.execute_and_fetch(query)  # type: ignore
        else:
            self.db.execute(query)
            return None
