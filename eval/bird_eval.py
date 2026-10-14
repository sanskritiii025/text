"""Score the agent on BIRD Mini-Dev (SQLite), the standard text-to-SQL benchmark.

Metric: execution accuracy (EX), BIRD's official metric -- a prediction is
correct when set(predicted rows) == set(gold rows), exactly as in BIRD's
evaluation_ex.py. Column names and row order do not matter; values do.

One run produces two scores, because every attempt is logged:
    first-try EX     : the model's first query only (self-correction off)
    self-corrected EX: after up to --retries fixes driven by error feedback

The run is resumable: results append to a JSONL file and finished questions
are skipped, so a free-tier daily quota can be spread over several days.

Usage:
    python eval/bird_eval.py --n 150                      # stratified sample, seed 42
    python eval/bird_eval.py --n 150 --no-evidence        # ablation: drop BIRD's hints
    python eval/bird_eval.py --report eval/results/<file>.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from agent.db import SQLiteDB       # noqa: E402
from agent.graph import build_agent  # noqa: E402
from agent.llm import build_llm, model_label  # noqa: E402

BIRD_DIR = ROOT / "data" / "bird" / "minidev" / "MINIDEV"
RESULTS_DIR = ROOT / "eval" / "results"


def load_questions():
    return json.loads((BIRD_DIR / "mini_dev_sqlite.json").read_text(encoding="utf-8"))


def stratified_sample(questions, n, seed):
    """Same difficulty mix as the full set, and the same sample every run."""
    if n >= len(questions):
        return questions
    rng = random.Random(seed)
    by_diff = defaultdict(list)
    for q in questions:
        by_diff[q["difficulty"]].append(q)
    picked = []
    for diff, items in sorted(by_diff.items()):
        k = round(n * len(items) / len(questions))
        picked += rng.sample(items, min(k, len(items)))
    return sorted(picked, key=lambda q: q["question_id"])


def db_path(db_id):
    return BIRD_DIR / "dev_databases" / db_id / f"{db_id}.sqlite"


def same_result(db, pred_sql, gold_rows):
    try:
        pred = db.run(pred_sql)
    except Exception:
        return False
    return set(map(tuple, pred.rows)) == gold_rows


def evaluate(args):
    questions = stratified_sample(load_questions(), args.n, args.seed)
    tag = f"{model_label()}_n{len(questions)}_r{args.retries}{'_noevidence' if args.no_evidence else ''}"
    out_path = RESULTS_DIR / (re.sub(r"[^A-Za-z0-9._-]+", "-", tag) + ".jsonl")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["question_id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l}
    print(f"{tag}: {len(questions)} questions, {len(done)} already done -> {out_path.name}")

    llm = build_llm(temperature=0)
    dbs, agents = {}, {}
    with out_path.open("a", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            if q["question_id"] in done:
                continue
            db_id = q["db_id"]
            if db_id not in agents:
                dbs[db_id] = SQLiteDB(str(db_path(db_id)))
                agents[db_id] = build_agent(dbs[db_id], glossary=None, max_retries=args.retries,
                                            answer=False, style="benchmark", default_limit=None,
                                            retry_on_empty=False, llm=llm)
            db = dbs[db_id]
            gold_rows = set(map(tuple, db.run(q["SQL"]).rows))
            evidence = [] if args.no_evidence or not q.get("evidence") else [q["evidence"]]

            start = time.time()
            try:
                state = agents[db_id].invoke({"question": q["question"], "evidence": evidence})
                attempts = state.get("attempts", [])
            except Exception as e:   # quota exhausted, network, etc. -- stop, resume later
                print(f"  stopped at question {q['question_id']}: {str(e).splitlines()[0][:200]}")
                break
            elapsed = time.time() - start

            first = attempts[0] if attempts else {"sql": "", "error": "no attempt"}
            final = attempts[-1] if attempts else first
            first_ok = not first["error"] and same_result(db, first["sql"], gold_rows)
            final_ok = not final["error"] and same_result(db, final["sql"], gold_rows)
            rec = {
                "question_id": q["question_id"], "db_id": db_id, "difficulty": q["difficulty"],
                "question": q["question"], "gold_sql": q["SQL"],
                "first_sql": first["sql"], "first_error": first["error"], "first_correct": first_ok,
                "final_sql": final["sql"], "final_error": final["error"], "final_correct": final_ok,
                "n_attempts": len(attempts), "latency_sec": round(elapsed, 2),
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            mark = "PASS" if final_ok else "FAIL"
            fix = " (fixed by retry)" if final_ok and not first_ok else ""
            print(f"  [{i}/{len(questions)}] {mark} {q['difficulty']:<11} {db_id:<24} {elapsed:5.1f}s{fix}")
            time.sleep(args.sleep)
    report(out_path)


def report(path: Path):
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l]
    if not recs:
        print("no results yet")
        return
    order = ["simple", "moderate", "challenging"]
    groups = defaultdict(list)
    for r in recs:
        groups[r["difficulty"]].append(r)

    def pct(rs, key):
        return 100 * sum(r[key] for r in rs) / len(rs)

    print(f"\nResults: {Path(path).name}  ({len(recs)} questions)")
    print(f"{'difficulty':<12}{'n':>5}{'first-try EX':>15}{'self-corrected EX':>20}")
    for d in order + ["total"]:
        rs = recs if d == "total" else groups.get(d, [])
        if rs:
            print(f"{d:<12}{len(rs):>5}{pct(rs, 'first_correct'):>14.1f}%{pct(rs, 'final_correct'):>19.1f}%")

    errors = [r for r in recs if r["first_error"]]
    fixed = [r for r in recs if r["final_correct"] and not r["first_correct"]]
    kinds = Counter(r["first_error"].split(":")[0][:40] for r in errors)
    print(f"\nFirst attempts that errored: {len(errors)}; turned correct by self-correction: {len(fixed)}")
    print("Most common first-attempt errors:", kinds.most_common(5))
    wrong_but_ran = sum(1 for r in recs if not r["final_error"] and not r["final_correct"])
    print(f"Ran without error but returned the wrong rows: {wrong_but_ran} "
          "(self-correction cannot catch these -- the database gives no error to learn from)")
    lat = sorted(r["latency_sec"] for r in recs)
    print(f"Latency: median {lat[len(lat) // 2]:.1f}s, p90 {lat[int(len(lat) * 0.9)]:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150, help="questions to sample (500 = full Mini-Dev)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retries", type=int, default=2, help="self-correction attempts after the first")
    p.add_argument("--no-evidence", action="store_true", help="ablation: withhold BIRD's hints")
    p.add_argument("--sleep", type=float, default=4.0, help="pause between questions (free-tier rate limits)")
    p.add_argument("--report", help="only print the report for an existing results file")
    a = p.parse_args()
    report(Path(a.report)) if a.report else evaluate(a)
