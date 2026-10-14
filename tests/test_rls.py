"""Try to break row-level security the way a bad generated query would.

Run: python tests/test_rls.py   (every "attack" row must end in BLOCKED or a
filtered count; if any attack returns Hartford data, RLS is broken).
"""
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def run(role, query):
    conn = psycopg2.connect(dbname=os.getenv("DB_NAME", "postgres"), user=role,
                            password=os.getenv("RX_ROLE_PASSWORD"),
                            host=os.getenv("DB_HOST", "localhost"), port=os.getenv("DB_PORT", "5432"),
                            sslmode=os.getenv("DB_SSLMODE", "prefer"))
    try:
        cur = conn.cursor()
        cur.execute(query)
        return cur.fetchall() if cur.description else "ok"
    except Exception as e:
        return "BLOCKED: " + str(e).splitlines()[0]
    finally:
        conn.close()


CASES = [
    ("rx_national", "SELECT count(*) FROM prescribers"),
    ("mgr_norwich", "SELECT count(*) FROM prescribers"),
    ("mgr_norwich", "SELECT DISTINCT territory_id FROM prescriptions"),
    ("mgr_norwich", "SELECT count(*) FROM prescribers WHERE territory_id = 'T-HARTFORD'"),
    ("mgr_norwich", "SELECT set_config('app.territory','T-HARTFORD',false), (SELECT count(*) FROM prescribers)"),
    ("mgr_norwich", "SET ROLE rx_national"),
    ("mgr_norwich", "SELECT * FROM user_territories"),
    ("mgr_norwich", "UPDATE drugs SET brand_name = 'x' WHERE drug_id = 1"),
    ("mgr_norwich", "SELECT count(*) FROM public.opportunities"),
    ("mgr_norwich", "SELECT pg_sleep(20)"),
]

if __name__ == "__main__":
    for role, query in CASES:
        print(f"{role:<12} | {query[:72]:<72} | {run(role, query)}")
