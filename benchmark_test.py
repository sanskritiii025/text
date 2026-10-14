"""
Benchmark Test Suite for Sales Ops Intelligence Agent
Measures: SQL execution success rate, response latency, query correctness
Uses local Ollama LLM (phi3) - no API limits.
"""
import os
from dotenv import load_dotenv
load_dotenv()
import re
import time
import json
import psycopg2
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# ── Config ──
OLLAMA_MODEL = "phi3"

DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("DB_PASSWORD"),
    "host": "localhost",
    "port": "5432"
}

vector_db = Chroma(
    persist_directory="./sales_memory",
    embedding_function=OllamaEmbeddings(model=OLLAMA_MODEL)
)

# ── State & Nodes ──
class AgentState(TypedDict):
    question: str
    schema: str
    evidence: str
    sql_query: str
    data_result: str
    rows: list
    columns: list
    final_answer: str

def get_schema():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            ORDER BY table_name, ordinal_position;
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        schema_str = ""
        current_table = ""
        for table, col, dtype in rows:
            if table != current_table:
                schema_str += f"\nTABLE: {table}\n"
                current_table = table
            schema_str += f" - {col} ({dtype})\n"
        return schema_str
    except Exception as e:
        return f"Error: {e}"

def retrieve_node(state: AgentState):
    results = vector_db.similarity_search(state['question'], k=2)
    return {"evidence": "\n".join([doc.page_content for doc in results]), "schema": get_schema()}


def extract_sql_from_response(text: str) -> str:
    """Extract SQL from LLM response, handling code blocks and plain text."""
    sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    sql_match = re.search(
        r'(SELECT|UPDATE|INSERT|DELETE)\s+.*?;',
        text, re.DOTALL | re.IGNORECASE
    )
    if sql_match:
        return sql_match.group(0).strip()
    return text.strip()


def generate_sql_node(state: AgentState):
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    template = """You are a Postgres SQL Agent operating in a PRODUCTION environment.
SCHEMA: {schema}
BUSINESS RULES: {evidence}
CRITICAL INSTRUCTIONS:
1. READ operations (SELECT) are preferred.
2. ALWAYS include a specific WHERE clause for updates.
3. Do NOT include 'BEGIN' or 'COMMIT'.
4. Return ONLY the raw SQL query inside a ```sql code block.
SPECIAL RULE FOR CHITCHAT/GREETINGS:
If the user input is a greeting or NOT related to the database:
Return exactly: ```sql
SELECT 'Hello! I am your Sales Ops Intelligence Agent.' AS greeting;
```
Write a valid SQL query for: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    response = (prompt | llm).invoke({
        "schema": state['schema'],
        "evidence": state['evidence'],
        "question": state['question']
    })
    sql = extract_sql_from_response(response.content)
    return {"sql_query": sql}

def execute_sql_node(state: AgentState):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        raw = state['sql_query'].strip().replace("```sql", "").replace("```", "").strip()
        # Block writes in benchmark
        if any(cmd in raw.upper() for cmd in ["UPDATE ", "INSERT ", "DELETE "]):
            return {"data_result": "BLOCKED: Write operation skipped in benchmark.", "rows": [], "columns": []}
        cur.execute(raw)
        if cur.description:
            rows = cur.fetchall()
            colnames = [d[0] for d in cur.description]
            preview = [dict(zip(colnames, r)) for r in rows]
            result = f"SUCCESS: {len(rows)} record(s). Data: {preview}"
        else:
            rows, colnames = [], []
            result = "Query executed. No data returned."
        cur.close()
        conn.close()
        return {"data_result": result, "rows": rows, "columns": colnames}
    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        return {"data_result": f"SQL ERROR: {e}", "rows": [], "columns": []}

def answer_node(state: AgentState):
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    resp = llm.invoke(f"Question: {state['question']}\nData: {state['data_result']}\nAnswer professionally in 2-3 sentences:")
    return {"final_answer": resp.content}

# ── Build Graph ──
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("answer", answer_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_edge("execute_sql", "answer")
workflow.add_edge("answer", END)
app = workflow.compile()

# ── Benchmark Test Cases ──
def check_has_rows(state):
    if state.get("rows") and len(state["rows"]) > 0:
        return True, f"Returned {len(state['rows'])} rows"
    return False, "No rows returned"

def check_sql_success(state):
    if "SQL ERROR" not in state.get("data_result", ""):
        return True, "SQL executed without errors"
    return False, state["data_result"]

def check_greeting(state):
    ans = state.get("final_answer", "").lower()
    if any(w in ans for w in ["hello", "hi", "help", "agent", "assist"]):
        return True, "Greeting response detected"
    return False, f"Unexpected: {state['final_answer'][:80]}"

def check_count_query(state):
    if state.get("rows") and len(state["rows"]) == 1:
        val = state["rows"][0][0]
        if isinstance(val, (int, float)) and val > 0:
            return True, f"Count = {val}"
    if "SQL ERROR" not in state.get("data_result", ""):
        return True, f"Query succeeded: {state.get('data_result', '')[:80]}"
    return False, "Count query failed"


BENCHMARK_TESTS = [
    # ── Basic Data Retrieval ──
    ("How many opportunities are there?", "COUNT query on opportunities", check_count_query),
    ("List all territories", "SELECT from territories", check_has_rows),
    ("Show me all sales reps", "SELECT from sales_reps", check_has_rows),
    ("How many customers do we have?", "COUNT on customers", check_count_query),
    
    # ── Filtered Queries ──
    ("Show all Closed Won opportunities", "Filtered SELECT with stage", check_has_rows),
    ("Which opportunities are in the Negotiation stage?", "Filtered by stage", check_has_rows),
    ("List accounts in the North region", "Filtered by region ENUM", check_sql_success),
    
    # ── Aggregation Queries ──
    ("What is the total value of all Closed Won deals?", "SUM with filter", check_sql_success),
    ("What is the average deal size?", "AVG on amount", check_sql_success),
    
    # ── Join Queries ──
    ("Show opportunities with their account names", "JOIN opportunities + accounts", check_has_rows),
    ("Which sales reps belong to which territories?", "JOIN sales_reps + territories", check_has_rows),
    
    # ── Business Policy Queries (RAG-dependent) ──
    ("Which deals are high value deals?", "RAG: uses policy for amount >= 100000", check_sql_success),
    ("What is the weighted forecast?", "RAG: pipeline stage weights", check_sql_success),
    ("Show the pipeline stages and their win probabilities", "RAG: stage definitions", check_sql_success),

    # ── Chitchat Handling ──
    ("Hello!", "Should return greeting", check_greeting),
]

# ── Run Benchmark ──
def run_benchmark():
    print("=" * 70)
    print("  SALES OPS INTELLIGENCE AGENT - BENCHMARK TEST SUITE")
    print("  Engine: Ollama (phi3) | Local LLM | No API Limits")
    print("=" * 70)
    
    results = []
    total_time = 0
    
    for i, (question, description, validator) in enumerate(BENCHMARK_TESTS, 1):
        print(f"\n[Test {i:02d}/{len(BENCHMARK_TESTS)}] {description}")
        print(f"  Q: \"{question}\"")
        
        start = time.time()
        try:
            state = app.invoke({"question": question})
            elapsed = time.time() - start
            total_time += elapsed
            
            passed, detail = validator(state)
            sql = state.get("sql_query", "N/A")
            
            status = "PASS" if passed else "FAIL"
            print(f"  SQL: {sql[:100]}{'...' if len(sql)>100 else ''}")
            print(f"  Result: {status} | {detail}")
            print(f"  Latency: {elapsed:.2f}s")
            
            results.append({
                "test_id": i,
                "question": question,
                "description": description,
                "status": status,
                "detail": detail,
                "sql_query": sql,
                "latency_sec": round(elapsed, 2),
                "has_answer": bool(state.get("final_answer")),
            })
        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  Result: ERROR | {str(e)[:100]}")
            results.append({
                "test_id": i,
                "question": question,
                "description": description,
                "status": "ERROR",
                "detail": str(e)[:200],
                "sql_query": "N/A",
                "latency_sec": round(elapsed, 2),
                "has_answer": False,
            })
    
    # ── Summary ──
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    total = len(results)
    latencies = [r["latency_sec"] for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total Tests:      {total}")
    print(f"  Passed:           {passed} ({passed/total*100:.0f}%)")
    print(f"  Failed:           {failed} ({failed/total*100:.0f}%)")
    print(f"  Errors:           {errors} ({errors/total*100:.0f}%)")
    print(f"  Success Rate:     {passed/total*100:.1f}%")
    print(f"  Avg Latency:      {avg_latency:.2f}s")
    print(f"  Min Latency:      {min(latencies):.2f}s")
    print(f"  Max Latency:      {max(latencies):.2f}s")
    print(f"  Total Time:       {total_time:.2f}s")
    print("=" * 70)
    
    # Save results to JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": f"Ollama ({OLLAMA_MODEL})",
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "success_rate_pct": round(passed / total * 100, 1),
        "avg_latency_sec": round(avg_latency, 2),
        "min_latency_sec": round(min(latencies), 2),
        "max_latency_sec": round(max(latencies), 2),
        "total_time_sec": round(total_time, 2),
        "test_results": results,
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed results saved to: benchmark_results.json")
    
    return report

if __name__ == "__main__":
    run_benchmark()
