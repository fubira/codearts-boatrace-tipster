"""DuckDB connection manager for analytical queries.

Uses DuckDB's SQLite scanner to read the existing SQLite database
without data migration. All tables are accessed via the 'db' schema.
"""

from pathlib import Path

import duckdb

# Project root (ml/../) resolved from this file's location
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DB_PATH = str(_PROJECT_ROOT / "data" / "boatrace-tipster.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB in-memory connection with SQLite file attached."""
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db_path}' AS db (TYPE sqlite, READ_ONLY)")
    return conn
