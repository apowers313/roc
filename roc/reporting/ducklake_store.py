"""DuckLake-based storage for ROC run data.

Wraps a single DuckDB connection with a DuckLake catalog (SQLite backend)
that stores all data as standard Parquet files. Handles both writes (INSERT)
and reads (SELECT) with automatic schema evolution.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import duckdb
import pandas as pd
import pyarrow as pa

# Counters are created lazily on first use to avoid a circular import:
# ``observability`` imports ``parquet_exporter`` which imports this module.
_conn_opened: Any = None
_conn_closed: Any = None


def _record_conn(opened: bool, read_only: bool) -> None:
    """Increment the open/close counter. Lazy-creates the OTel instruments."""
    global _conn_opened, _conn_closed
    try:
        if _conn_opened is None:
            from roc.reporting.observability import Observability

            _conn_opened = Observability.meter.create_counter(
                "roc.ducklake.connections_opened",
                description="DuckLake connections opened; label mode=read|write",
            )
            _conn_closed = Observability.meter.create_counter(
                "roc.ducklake.connections_closed",
                description="DuckLake connections closed; label mode=read|write",
            )
        counter = _conn_opened if opened else _conn_closed
        counter.add(1, {"mode": "read" if read_only else "write"})
    except Exception:
        pass


class DuckLakeStore:
    """Unified DuckLake read/write store for a single run.

    Owns a single DuckDB connection with a DuckLake catalog attached.
    All access is serialized through ``_lock`` since DuckDB connections
    are not thread-safe.
    """

    TABLES = ("screens", "saliency", "events", "logs", "metrics")

    def __init__(
        self,
        run_dir: Path,
        *,
        read_only: bool = False,
        alias: str = "lake",
    ) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_path = run_dir / "catalog.duckdb"
        self._data_path = run_dir / "data"
        self._lock = threading.Lock()
        self._alias = alias

        self._read_only = read_only
        self._closed = False
        config: dict[str, Any] = {}
        if read_only:
            config["threads"] = 1
        self._conn = duckdb.connect(config=config)
        self._conn.execute("INSTALL ducklake;")
        self._conn.execute("LOAD ducklake;")
        catalog = str(self._catalog_path).replace("'", "''")
        data = str(self._data_path).replace("'", "''")
        self._conn.execute(f"ATTACH 'ducklake:{catalog}' AS {alias} (DATA_PATH '{data}')")
        _record_conn(opened=True, read_only=read_only)
        try:
            self._conn.execute("SELECT 1").fetchone()
        except Exception:
            logger.error("DuckLake connection validation failed for %s", run_dir)
            raise
        if not read_only:
            # Enable data inlining: small inserts (< 500 rows) are stored
            # in the catalog rather than creating individual parquet files.
            # Periodic CHECKPOINT calls flush inlined data to parquet and
            # merge small files.  This avoids the "7000 tiny files" problem
            # that caused 5-11s query times with raw parquet reads.
            self._conn.execute(f"CALL {alias}.set_option('data_inlining_row_limit', '500')")

        self._known_columns: dict[str, list[str]] = {}

    # -- Write path --

    def insert(self, table: str, records: list[dict[str, Any]]) -> None:
        """Insert records into a DuckLake table, creating it if needed.

        Handles schema evolution: new columns in *records* are added via
        ``ALTER TABLE ADD COLUMN`` before the INSERT.
        """
        if not records:
            return

        arrow_table = pa.Table.from_pylist(records)

        with self._lock:
            if table not in self._known_columns:
                if not self._table_exists(table):
                    self._create_table(table, arrow_table.schema)
                else:
                    self._known_columns[table] = self._get_columns(table)

            # Evolve schema if needed
            existing_cols = set(self._known_columns.get(table, []))
            for field in arrow_table.schema:
                if field.name not in existing_cols:
                    duck_type = self._arrow_to_duck_type(field.type)
                    self._conn.execute(
                        f'ALTER TABLE {self._alias}."{table}" ADD COLUMN "{field.name}" {duck_type}'
                    )
            # Refresh known columns after any evolution
            self._known_columns[table] = self._get_columns(table)

            # Insert via arrow scan
            self._conn.register("_arrow_batch", arrow_table)
            try:
                self._conn.execute(
                    f'INSERT INTO {self._alias}."{table}" BY NAME SELECT * FROM _arrow_batch'  # nosec B608
                )
            finally:
                self._conn.unregister("_arrow_batch")

    # -- Read path --

    def execute(self, sql: str, params: list[Any] | None = None) -> duckdb.DuckDBPyConnection:
        """Execute a SQL query against the lake.

        .. warning:: The returned cursor is only valid while no other
           thread uses this connection.  Prefer ``query_df`` / ``query_one``
           for thread-safe reads.
        """
        with self._lock:
            if params:
                return self._conn.execute(sql, params)
            return self._conn.execute(sql)

    def query_df(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Execute a query and return a DataFrame, holding the lock for the full cycle."""
        with self._lock:
            if params:
                return self._conn.execute(sql, params).fetchdf()
            return self._conn.execute(sql).fetchdf()

    def query_one(
        self,
        sql: str,
        params: list[Any] | None = None,
    ) -> tuple[Any, ...] | None:
        """Execute a query and return a single row, holding the lock for the full cycle."""
        with self._lock:
            if params:
                return self._conn.execute(sql, params).fetchone()
            return self._conn.execute(sql).fetchone()

    def query_step_batch(
        self,
        steps: list[int],
        tables: list[str] | None = None,
    ) -> dict[int, dict[str, pd.DataFrame]]:
        """Query multiple tables for one or more steps efficiently.

        Executes one ``SELECT ... WHERE step IN (...)`` per table under a
        single lock acquisition, then splits the results by step.  This
        avoids the per-query DuckLake catalog overhead that makes sequential
        ``get_step`` calls slow (~100ms each -> 500ms for 5 tables).

        Returns ``{step: {table_name: DataFrame}}``.
        """
        if tables is None:
            tables = list(self.TABLES)
        results: dict[int, dict[str, pd.DataFrame]] = {
            s: {t: pd.DataFrame() for t in tables} for s in steps
        }
        if not steps:
            return results
        step_list = ", ".join(str(int(s)) for s in steps)
        with self._lock:
            for table in tables:
                self._query_table_for_steps(table, step_list, steps, results)
        return results

    def _query_table_for_steps(
        self,
        table: str,
        step_list: str,
        steps: list[int],
        results: dict[int, dict[str, pd.DataFrame]],
    ) -> None:
        """Query a single table for the given steps and split results (must hold lock)."""
        try:
            if not self._table_exists(table):
                return
            src = f'{self._alias}."{table}"'
            df = self._conn.execute(f"SELECT * FROM {src} WHERE step IN ({step_list})").fetchdf()  # nosec B608
            if len(df) == 0 or "step" not in df.columns:
                return
            for s in steps:
                mask = df["step"] == s
                if mask.any():
                    results[s][table] = df[mask].reset_index(drop=True)
        except Exception:
            logger.debug("table query failed for %s steps=%s", table, steps, exc_info=True)

    def has_table(self, table: str) -> bool:
        """Check whether a table exists in the DuckLake catalog."""
        with self._lock:
            return self._table_exists(table)

    @staticmethod
    def is_valid_run(run_dir: Path) -> bool:
        """Check if a directory contains a valid DuckLake catalog."""
        return (run_dir / "catalog.duckdb").exists() or (run_dir / "catalog.sqlite").exists()

    def checkpoint(self) -> None:
        """Run DuckLake maintenance: flush inlined data, merge small files."""
        with self._lock:
            self._conn.execute(f"CHECKPOINT {self._alias}")

    def close(self) -> None:
        """Detach the catalog and close the connection."""
        with self._lock:
            if self._closed:
                return
            try:
                self._conn.execute(f"DETACH {self._alias}")
            except duckdb.Error:
                pass
            self._conn.close()
            self._closed = True
            _record_conn(opened=False, read_only=self._read_only)

    # -- Internal helpers --

    def _table_exists(self, table: str) -> bool:
        """Check table existence (must hold lock)."""
        try:
            self._conn.execute(f'SELECT 1 FROM {self._alias}."{table}" LIMIT 0')  # nosec B608
            return True
        except duckdb.CatalogException:
            return False

    def _create_table(self, table: str, schema: pa.Schema) -> None:
        """Create a new DuckLake table from an Arrow schema (must hold lock)."""
        cols = []
        for field in schema:
            duck_type = self._arrow_to_duck_type(field.type)
            cols.append(f'"{field.name}" {duck_type}')
        col_defs = ", ".join(cols)
        self._conn.execute(f'CREATE TABLE {self._alias}."{table}" ({col_defs})')
        self._known_columns[table] = [f.name for f in schema]

    def _get_columns(self, table: str) -> list[str]:
        """Get column names for a table (must hold lock)."""
        result = self._conn.execute(
            f"SELECT column_name FROM information_schema.columns "  # nosec B608
            f"WHERE table_schema = 'main' AND table_name = '{table}'"
        ).fetchall()
        return [row[0] for row in result]

    @staticmethod
    def _arrow_to_duck_type(arrow_type: pa.DataType) -> str:
        """Map an Arrow type to a DuckDB SQL type string."""
        if pa.types.is_int64(arrow_type):
            return "BIGINT"
        if pa.types.is_int32(arrow_type):
            return "INTEGER"
        if pa.types.is_float64(arrow_type):
            return "DOUBLE"
        if pa.types.is_float32(arrow_type):
            return "FLOAT"
        if pa.types.is_boolean(arrow_type):
            return "BOOLEAN"
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return "VARCHAR"
        if pa.types.is_null(arrow_type):
            return "VARCHAR"
        # Fallback
        return "VARCHAR"
