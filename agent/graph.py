"""The agent: a LangGraph state machine.

    retrieve -> generate -> guard --ok--> execute --ok--> answer -> END
                   ^          |              |
                   |        error          error
                   +---- (retry, max N) -----+

Self-correction: when the guard rejects a query or the database raises an
error, the error message goes back to the model along with its own failed
query, and it gets another try. Most text-to-SQL failures are small
(a misspelled column, a missing GROUP BY) and the database error tells the
model exactly what to fix.

The same graph runs the app (Postgres + glossary + answer) and the BIRD
benchmark (SQLite + BIRD's evidence + no answer step).
"""
import re
from typing import Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent.guard import check_sql
from agent.llm import build_llm


class SQLGeneration(BaseModel):
    """Structured contract for the model's output. Reasoning comes first so
    the model thinks through joins and filters before committing to SQL."""
    reasoning: str = Field(description="Brief plan: which tables, joins, filters and aggregation answer the question.")
    sql: str = Field(description="One read-only SQL query. No markdown fences, no trailing explanation.")


class AgentState(TypedDict, total=False):
    question: str
    history: list          # earlier (question, sql) pairs, for follow-up questions
    evidence: list         # glossary entries or benchmark hints
    reasoning: str
    sql: str
    error: str             # last guard/database error, fed back on retry
    attempts: list         # every attempt: {"sql", "error"} -- shown in the UI and scored in eval
    result: object         # QueryResult
    answer: str


DIALECT_RULES = {
    "postgres": (
        "- PostgreSQL. Tables are on the search path: write prescriptions, not rx.prescriptions.\n"
        "- Match text case-insensitively (lower(col) LIKE 'eliquis%').\n"
        "- Integer division truncates: multiply by 100.0 or cast before dividing.\n"
        "- Use NULLIF(x, 0) only on a denominator. Never on a numerator or a value you filter on: it turns 0 into NULL and silently drops those rows.\n"
        "- ROUND(x, n) needs a numeric argument: ROUND((expr)::numeric, 1)."
    ),
    "sqlite": (
        "- SQLite. Quote identifiers that contain spaces or symbols with double quotes or backticks.\n"
        "- Integer division truncates: CAST(x AS REAL) before dividing."
    ),
}

STYLE_RULES = {
    # App users want readable output; extra context columns help them.
    "analyst": (
        "- When listing prescribers or drugs, include readable columns (prescriber_name, specialty, city, brand_name), not only IDs.\n"
        "- Round money to 0 decimals and percentages to 1 decimal.\n"
        "- For 'top N' questions use ORDER BY ... LIMIT N. Default to top 10 when N is not given."
    ),
    # Benchmark scoring compares result sets exactly; extra columns count as wrong.
    "benchmark": (
        "- Select exactly the columns the question asks for, in the order asked, and nothing else.\n"
        "- Do not round unless the question asks for it."
    ),
}

GENERATE_PROMPT = """You are an expert data analyst who writes correct SQL.

DATABASE SCHEMA:
{schema}

BUSINESS DEFINITIONS AND HINTS (follow these exactly when relevant):
{evidence}

RULES:
- Write exactly one read-only SELECT query (WITH ... SELECT is fine).
- Use only tables and columns that appear in the schema.
{dialect_rules}
{style_rules}
{history}{feedback}
QUESTION: {question}"""

ANSWER_PROMPT = """You are a commercial analytics assistant for a pharma sales team.
Answer the user's question in 2-4 sentences using ONLY the query result below.
Quote the key numbers. Do not invent numbers or causes that are not in the data.
Always state the assumptions the SQL made, read from the SQL itself: the year(s) covered and,
if a market or drug group was involved, which drugs or ATC codes defined it
(e.g. "2023, anticoagulants = Eliquis, Xarelto, Savaysa, Pradaxa").
If the result is empty, say no matching data was found in the territories this user can see.
If the definitions below contain a caveat relevant to this question (e.g. NRx not available,
patient counts cannot be summed, Medicare-only data), mention it in one short sentence.

DEFINITIONS:
{evidence}

QUESTION: {question}
SQL USED: {sql}
RESULT ({n_rows} rows{truncated}):
{table}"""


EMPTY_RESULT = "Query ran but returned 0 rows."
EMPTY_HINT = (" Check every filter against the schema comments and business definitions: exact text values,"
              " brand-name matching, ATC codes instead of keyword searches on class names. If you are sure the"
              " filters are right, return the same query unchanged.")


def _extract_sql(text: str) -> str:
    """Pull SQL out of model text: a markdown fence with any language tag, or prose around a query."""
    m = re.search(r"```(.*?)```", text, re.DOTALL)
    if m:
        # Drop the fence's language tag. Matching only "sql" once left "ite" from ```sqlite.
        return re.sub(r"^\s*(?:sql\w*|postgres\w*|psql|mysql)\b", "", m.group(1), flags=re.IGNORECASE).strip()
    m = re.search(r"\b(WITH|SELECT)\b.*", text, re.DOTALL | re.IGNORECASE)
    return m.group(0).strip() if m else text.strip()


def _markdown_table(result, max_rows: int = 25) -> str:
    if not result.columns:
        return "(no columns)"
    lines = ["| " + " | ".join(result.columns) + " |",
             "|" + "---|" * len(result.columns)]
    for row in result.rows[:max_rows]:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def build_agent(db, glossary=None, *, max_retries: int = 2, answer: bool = True,
                style: str = "analyst", default_limit: Optional[int] = 500,
                retry_on_empty: bool = True, llm=None):
    """Compile the graph for one database connection.

    glossary: a Glossary to retrieve from, or None to use state["evidence"] as given (benchmark).
    max_retries: self-correction attempts after the first; 0 disables self-correction.
    retry_on_empty: treat a 0-row result as suspicious and re-check filters once.
    """
    llm = llm or build_llm(temperature=0)
    schema = db.schema_text()
    allowed_tables = db.tables()
    prompt = ChatPromptTemplate.from_template(GENERATE_PROMPT)

    def retrieve(state: AgentState):
        if glossary is None:
            return {"evidence": state.get("evidence", []), "attempts": []}
        return {"evidence": glossary.search(state["question"], k=4), "attempts": []}

    def generate(state: AgentState):
        history = ""
        if state.get("history"):
            pairs = "\n".join(f"Q: {q}\nSQL: {s}" for q, s in state["history"][-3:])
            history = f"\nEARLIER IN THIS CONVERSATION (the question may be a follow-up):\n{pairs}\n"
        feedback = ""
        if state.get("error"):
            feedback = (f"\nYOUR PREVIOUS QUERY FAILED.\nQuery:\n{state['sql']}\nError:\n{state['error']}\n"
                        "Write a corrected query that fixes this error.\n")
        inputs = {
            "schema": schema,
            "evidence": "\n\n".join(state.get("evidence") or []) or "(none)",
            "dialect_rules": DIALECT_RULES[db.dialect],
            "style_rules": STYLE_RULES[style],
            "history": history,
            "feedback": feedback,
            "question": state["question"],
        }
        try:
            out = (prompt | llm.with_structured_output(SQLGeneration)).invoke(inputs)
            if out and out.sql.strip():
                return {"sql": _extract_sql(out.sql), "reasoning": out.reasoning, "error": ""}
        except Exception:
            pass   # some models emit malformed tool-call JSON; fall back to plain text
        text = (prompt | llm).invoke(inputs).content
        if isinstance(text, list):   # some providers return content blocks
            text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
        return {"sql": _extract_sql(text), "reasoning": "", "error": ""}

    def guard(state: AgentState):
        g = check_sql(state["sql"], db.dialect, allowed_tables, default_limit)
        attempts = state.get("attempts", []) + [{"sql": g.sql, "error": g.error}]
        return {"sql": g.sql, "error": g.error, "attempts": attempts}

    def execute(state: AgentState):
        try:
            result = db.run(state["sql"])
            # An empty result is the most common *silent* failure: a text filter that
            # matches nothing ('anticoagulant' vs 'Direct factor Xa inhibitors') runs
            # fine and returns zero rows. Ask the model to re-check its filters once.
            # If it stands by the query, the empty result is accepted as real
            # (e.g. row-level security hid another territory's data).
            already_checked = any(a["error"] == EMPTY_RESULT for a in state["attempts"])
            retries_left = len(state["attempts"]) <= max_retries
            if retry_on_empty and not result.rows and not already_checked and retries_left:
                attempts = list(state["attempts"])
                attempts[-1] = {**attempts[-1], "error": EMPTY_RESULT}
                return {"result": None, "error": EMPTY_RESULT + EMPTY_HINT, "attempts": attempts}
            return {"result": result, "error": ""}
        except Exception as e:
            msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            attempts = list(state["attempts"])
            attempts[-1] = {**attempts[-1], "error": msg}
            return {"result": None, "error": msg, "attempts": attempts}

    def write_answer(state: AgentState):
        result = state["result"]
        text = llm.invoke(ANSWER_PROMPT.format(
            evidence="\n\n".join(state.get("evidence") or []) or "(none)",
            question=state["question"], sql=state["sql"],
            n_rows=len(result.rows), truncated=", truncated" if result.truncated else "",
            table=_markdown_table(result),
        )).content
        if isinstance(text, list):
            text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
        return {"answer": text}

    def route(state: AgentState):
        if not state.get("error"):
            return "ok"
        return "retry" if len(state["attempts"]) <= max_retries else "give_up"

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("guard", guard)
    g.add_node("execute", execute)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "guard")
    g.add_conditional_edges("guard", route, {"ok": "execute", "retry": "generate", "give_up": END})
    if answer:
        g.add_node("answer", write_answer)
        g.add_conditional_edges("execute", route, {"ok": "answer", "retry": "generate", "give_up": END})
        g.add_edge("answer", END)
    else:
        g.add_conditional_edges("execute", route, {"ok": END, "retry": "generate", "give_up": END})
    return g.compile()
