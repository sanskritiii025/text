"""Ask the agent a question from the terminal.

Usage: python scripts/ask.py mgr_norwich "What was Eliquis's market share in 2023?"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agent.db import PostgresDB
from agent.graph import build_agent
from agent.knowledge import Glossary

role, question = sys.argv[1], " ".join(sys.argv[2:])
db = PostgresDB(role)
agent = build_agent(db, Glossary())
start = time.time()
state = agent.invoke({"question": question})
for i, a in enumerate(state["attempts"], 1):
    print(f"--- attempt {i}{' FAILED: ' + a['error'] if a['error'] else ''}\n{a['sql']}")
if state.get("result") is not None:
    r = state["result"]
    print(f"--- {len(r.rows)} rows: {r.columns}")
    for row in r.rows[:8]:
        print("   ", row)
print("--- definitions:", [e.splitlines()[0] for e in state.get("evidence", [])])
print("--- answer:\n" + (state.get("answer") or "(none)"))
print(f"--- {time.time() - start:.1f}s")
