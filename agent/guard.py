"""Static checks on generated SQL, before it touches the database.

The old version decided "is this a write?" by searching the text for
"UPDATE " etc. That misses "update\\n" and flags the word inside a string
literal. Parsing into a syntax tree with sqlglot answers the question
structurally instead.

This is defence in depth, not the security boundary: the database role is
already read-only and row-level secured. The guard's job is to reject bad
SQL early with an error message the LLM can use to fix its own query.
"""
import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

logging.getLogger("sqlglot").setLevel(logging.ERROR)   # we report parse problems ourselves

# Anything that changes data, schema, session state or privileges.
FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create,
             exp.Alter, exp.TruncateTable, exp.Grant, exp.Command, exp.Set, exp.Transaction,
             exp.Commit, exp.Rollback, exp.Use, exp.Pragma)
# Functions that can change session state or stall the server even inside a SELECT.
FORBIDDEN_FUNCTIONS = {"set_config", "pg_sleep", "pg_terminate_backend", "pg_cancel_backend",
                       "pg_read_file", "pg_read_binary_file", "lo_import", "dblink"}


@dataclass
class GuardResult:
    ok: bool
    sql: str            # possibly rewritten (e.g. LIMIT added)
    error: str = ""


def check_sql(sql: str, dialect: str, allowed_tables: set, default_limit: int | None = None) -> GuardResult:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return GuardResult(False, sql, "Empty query.")

    read = "postgres" if dialect == "postgres" else "sqlite"
    try:
        statements = [s for s in sqlglot.parse(sql, read=read) if s is not None]
    except sqlglot.errors.ParseError as e:
        return GuardResult(False, sql, f"SQL syntax error: {str(e).splitlines()[0]}")

    if len(statements) != 1:
        return GuardResult(False, sql, "Exactly one statement is allowed; do not chain statements with ';'.")
    tree = statements[0]

    if not isinstance(tree, exp.Query):
        return GuardResult(False, sql, f"Only read-only SELECT queries are allowed, got {tree.key.upper()}.")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN):
            return GuardResult(False, sql, f"{node.key.upper()} is not allowed; queries must be read-only.")
        if isinstance(node, (exp.Anonymous, exp.Func)):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS:
                return GuardResult(False, sql, f"Function {name}() is not allowed.")

    # Tables must exist. CTE names are defined inside the query, so they are fine too.
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables} | ctes
    unknown = sorted({t.name for t in tree.find_all(exp.Table)
                      if t.name and t.name.lower() not in allowed})
    if unknown:
        return GuardResult(False, sql, f"Unknown table(s): {', '.join(unknown)}. "
                                       f"Available: {', '.join(sorted(allowed_tables))}.")

    # Cap result size for interactive use (the benchmark passes None to keep exact results).
    # Appended as text rather than re-generated from the tree, so dialect-specific
    # syntax the model wrote (FILTER, ::casts) reaches the database untouched.
    if default_limit and isinstance(tree, exp.Select) and not tree.args.get("limit"):
        sql = f"{sql}\nLIMIT {default_limit}"

    return GuardResult(True, sql)
