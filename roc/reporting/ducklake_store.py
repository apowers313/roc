"""DuckLake-based storage for ROC run data.

Wraps a single DuckDB connection with a DuckLake catalog (SQLite backend)
that stores all data as standard Parquet files. Handles both writes (INSERT)
and reads (SELECT) with automatic schema evolution.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa


class DuckLakeStore:
    """Unified DuckLake read/write store for a single run.

    Owns a single DuckDB connection with a DuckLake catalog attached.
    All access is serialized through ``_lock`` since DuckDB connections
    are not thread-safe.
    """

    TABLES = ("screens", "saliency", "events", "logs", "metrics")

    def __init__(self, run_dir: Path, *, read_only: bool = False) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_path = run_dir / "catalog.sqlite"
        self._data_path = run_dir / "data"
        self._lock = threading.Lock()

        self._conn = duckdb.connect()
        self._conn.execute("INSTALL ducklake; INSTALL sqlite;")
        self._conn.execute("LOAD ducklake; LOAD sqlite;")
        catalog = str(self._catalog_path).replace("'", "''")
        data = str(self._data_path).replace("'", "''")
        self._conn.execute(f"ATTACH 'ducklake:sqlite:{catalog}' AS lake (DATA_PATH '{data}')")
        if not read_only:
            # Always write to Parquet (no data inlining in SQLite catalog)
            self._conn.execute("CALL lake.set_option('data_inlining_row_limit', '0')")

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
                        f'ALTER TABLE lake."{table}" ADD COLUMN "{field.name}" {duck_type}'
                    )
            # Refresh known columns after any evolution
            self._known_columns[table] = self._get_columns(table)

            # Insert via arrow scan
            self._conn.register("_arrow_batch", arrow_table)
            try:
                self._conn.execute(f'INSERT INTO lake."{table}" BY NAME SELECT * FROM _arrow_batch')
            finally:
                self._conn.unregister("_arrow_batch")

    # -- Read path --

    def execute(self, sql: str, params: list[Any] | None = None) -> duckdb.DuckDBPyConnection:
        """Execute a SQL query against the lake.

        The caller is responsible for calling ``.fetchdf()`` / ``.fetchone()``
        on the returned cursor.
        """
        with self._lock:
            if params:
                return self._conn.execute(sql, params)
            return self._conn.execute(sql)

    def has_table(self, table: str) -> bool:
        """Check whether a table exists in the DuckLake catalog."""
        with self._lock:
            return self._table_exists(table)

    @staticmethod
    def is_valid_run(run_dir: Path) -> bool:
        """Check if a directory contains a valid DuckLake catalog."""
        return (run_dir / "catalog.sqlite").exists()

    def close(self) -> None:
        """Detach the catalog and close the connection."""
        with self._lock:
            try:
                self._conn.execute("DETACH lake")
            except duckdb.Error:
                pass
            self._conn.close()

    # -- Internal helpers --

    def _table_exists(self, table: str) -> bool:
        """Check table existence (must hold lock)."""
        try:
            self._conn.execute(f'SELECT 1 FROM lake."{table}" LIMIT 0')
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
        self._conn.execute(f'CREATE TABLE lake."{table}" ({col_defs})')
        self._known_columns[table] = [f.name for f in schema]

    def _get_columns(self, table: str) -> list[str]:
        """Get column names for a table (must hold lock)."""
        result = self._conn.execute(
            f"SELECT column_name FROM information_schema.columns "
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
