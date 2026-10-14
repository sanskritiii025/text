"""Database adapters.

The agent talks to anything with this small interface:
    dialect          "postgres" | "sqlite"  (for the prompt and the SQL parser)
    tables()         names the SQL guard allows
    schema_text()    the schema description put in the prompt
    run(sql)         -> QueryResult, raises on any database error

PostgresDB serves the app: it logs in as the user's own role, so row-level
security applies to every query. SQLiteDB serves the BIRD benchmark, which
ships its databases as SQLite files. Same agent, two backends.
"""
import os
import sqlite3
import time
from dataclasses import dataclass

import psycopg2


@dataclass
class QueryResult:
    columns: list
    rows: list
    truncated: bool = False
    elapsed_sec: float = 0.0


class PostgresDB:
    dialect = "postgres"
    SCHEMA = "rx"
    HIDDEN_TABLES = {"user_territories"}   # access plumbing, not analytics data

    def __init__(self, role: str, password: str | None = None):
        self.role = role
        self._conn_args = {
            "dbname": os.getenv("DB_NAME", "postgres"),
            "user": role,
            "password": password or os.getenv("RX_ROLE_PASSWORD"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "sslmode": os.getenv("DB_SSLMODE", "prefer"),
        }
        self._schema_cache = None

    def _connect(self):
        return psycopg2.connect(**self._conn_args)

    def run(self, sql: str, max_rows: int = 500) -> QueryResult:
        start = time.time()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description is None:
                    return QueryResult([], [], elapsed_sec=time.time() - start)
                rows = cur.fetchmany(max_rows + 1)
                return QueryResult(
                    columns=[d[0] for d in cur.description],
                    rows=rows[:max_rows],
                    truncated=len(rows) > max_rows,
                    elapsed_sec=time.time() - start,
                )
        finally:
            conn.rollback()   # the role is read-only anyway; never leave a transaction open
            conn.close()

    def tables(self) -> set:
        cols = self._columns()
        return {t for t, *_ in cols}

    def _columns(self):
        if self._schema_cache is None:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT c.table_name, c.column_name, c.data_type,
                               col_description(format('%%I.%%I', c.table_schema, c.table_name)::regclass,
                                               c.ordinal_position),
                               obj_description(format('%%I.%%I', c.table_schema, c.table_name)::regclass)
                        FROM information_schema.columns c
                        WHERE c.table_schema = %s
                        ORDER BY c.table_name, c.ordinal_position
                    """, (self.SCHEMA,))
                    self._schema_cache = [r for r in cur.fetchall() if r[0] not in self.HIDDEN_TABLES]
            finally:
                conn.close()
        return self._schema_cache

    def schema_text(self) -> str:
        out, current = [], None
        for table, col, dtype, col_doc, table_doc in self._columns():
            if table != current:
                out.append(f"\nTABLE {table}" + (f"  -- {table_doc}" if table_doc else ""))
                current = table
            out.append(f"  {col} {dtype}" + (f"  -- {col_doc}" if col_doc else ""))
        return "\n".join(out).strip()

    def visible_territories(self) -> list:
        return self.run("SELECT territory_id FROM user_territories ORDER BY 1").rows


class SQLiteDB:
    dialect = "sqlite"

    def __init__(self, path: str, timeout_sec: float = 30.0, sample_rows: int = 3):
        self.path = path
        self.timeout_sec = timeout_sec
        self.sample_rows = sample_rows
        self._schema = None

    def _connect(self):
        # Read-only URI: the benchmark databases can never be modified.
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def run(self, sql: str, max_rows: int = 100000) -> QueryResult:
        start = time.time()
        conn = self._connect()
        # SQLite has no statement_timeout; abort from the progress handler instead.
        conn.set_progress_handler(lambda: int(time.time() - start > self.timeout_sec), 10000)
        try:
            cur = conn.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
            return QueryResult(
                columns=[d[0] for d in (cur.description or [])],
                rows=rows[:max_rows],
                truncated=len(rows) > max_rows,
                elapsed_sec=time.time() - start,
            )
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                raise TimeoutError(f"query exceeded {self.timeout_sec}s") from e
            raise
        finally:
            conn.close()

    def tables(self) -> set:
        conn = self._connect()
        try:
            return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()

    def schema_text(self) -> str:
        if self._schema is None:
            conn = self._connect()
            try:
                parts = []
                for name, ddl in conn.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
                    parts.append(ddl.strip())
                    if self.sample_rows:
                        cur = conn.execute(f'SELECT * FROM "{name}" LIMIT {self.sample_rows}')
                        cols = [d[0] for d in cur.description]
                        rows = cur.fetchall()
                        if rows:
                            parts.append(f"/* {self.sample_rows} example rows from {name}:\n"
                                         + " | ".join(cols) + "\n"
                                         + "\n".join(" | ".join(str(v)[:40] for v in r) for r in rows)
                                         + "\n*/")
                self._schema = "\n\n".join(parts)
            finally:
                conn.close()
        return self._schema
