"""Unit tests for the SQL guard. Run: python -m pytest tests/test_guard.py  (or python tests/test_guard.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.guard import check_sql

TABLES = {"prescriptions", "prescribers", "drugs", "territories"}

CASES = [
    # (sql, should_pass)
    ("SELECT count(*) FROM prescribers", True),
    ("select brand_name from drugs where brand_name = 'UPDATE me'", True),       # keyword inside a string
    ("WITH t AS (SELECT npi FROM prescriptions) SELECT count(*) FROM t", True),   # CTE name is not a table
    ("SELECT SUM(total_claims) FILTER (WHERE year = 2023) FROM prescriptions", True),
    ("SELECT 1 UNION SELECT 2", True),
    ("update drugs\nset brand_name='x'", False),                                  # lowercase + newline
    ("DELETE FROM prescribers", False),
    ("SELECT 1; DROP TABLE drugs", False),                                        # stacked statements
    ("SELECT set_config('app.territory','T-HARTFORD',false)", False),
    ("SELECT pg_sleep(30)", False),
    ("SELECT * FROM salaries", False),                                            # hallucinated table
    ("SET ROLE rx_national", False),
    ("SELEC * FROM drugs", False),                                                # syntax error
]


def test_guard():
    for sql, should_pass in CASES:
        result = check_sql(sql, "postgres", TABLES)
        assert result.ok == should_pass, f"{sql!r}: {result.error}"


if __name__ == "__main__":
    for sql, should_pass in CASES:
        r = check_sql(sql, "postgres", TABLES, default_limit=500)
        mark = "OK  " if r.ok == should_pass else "FAIL"
        print(f"{mark} {'pass' if r.ok else 'block'}  {sql[:60]!r:<64} {r.error}")
