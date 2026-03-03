"""DuckDB connection manager for analytical queries.

Uses DuckDB's SQLite scanner to read the existing SQLite database
without data migration. All tables are accessed via the 'db' schema.
"""

import duckdb


def get_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB in-memory connection with SQLite file attached."""
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db_path}' AS db (TYPE sqlite, READ_ONLY)")
    return conn
