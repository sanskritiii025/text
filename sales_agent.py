import os
import re
import time
import psycopg2
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.status import Status

console = Console()

# ── Config ──
# Provider is swappable at runtime: LLM_PROVIDER=ollama|openai|google
# The graph itself never changes -- every provider implements the same
# LangChain chat interface, so only this factory knows which one is live.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME", "postgres"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
}

if not DB_CONFIG["password"]:
    raise SystemExit(
        "DB_PASSWORD is not set. Copy .env.example to .env and fill it in."
    )


def build_llm(temperature: float = 0):
    """Return a chat model for the configured provider.

    Swapping GPT-4 in for the local model is one environment variable --
    no node in the graph needs to know which provider is behind it.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=temperature,
        )
    if LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=temperature,
        )
    return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)


def build_embeddings():
    """Embeddings for the policy vector store.

    NOTE: changing this invalidates ./sales_memory -- the query and the
    stored documents must be embedded by the same model to be comparable.
    Re-run setup_knowledge.py after switching.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        )
    return OllamaEmbeddings(model=os.getenv("EMBED_MODEL", OLLAMA_MODEL))


vector_db = Chroma(
    persist_directory="./sales_memory",
    embedding_function=build_embeddings(),
)

class SQLGeneration(BaseModel):
    """Structured contract for what generate_sql must return.

    Using a schema instead of parsing free text means the model is
    constrained to emit these three fields, and Pydantic validates them
    before anything reaches the database.
    """

    query: str = Field(
        description="A single valid PostgreSQL statement. No markdown fences, no BEGIN/COMMIT."
    )
    is_write: bool = Field(
        description="True if this statement modifies data (UPDATE, INSERT, DELETE)."
    )
    reasoning: str = Field(
        description="One sentence explaining how this query answers the question."
    )


class AgentState(TypedDict):
    question: str
    schema: str
    evidence: str
    sql_query: str
    sql_reasoning: str   # why the model wrote this query -- fed to the audit log
    model_flagged_write: bool  # the model's own claim; never trusted alone
    extraction_mode: str # "structured" or "regex-fallback", for observability
    data_result: str
    rows: list
    columns: list
    final_answer: str


def get_schema():
    """Dynamically fetches schema from Postgres."""
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
        return f"Error fetching schema: {str(e)}"


def retrieve_node(state: AgentState):
    query = state['question']
    results = vector_db.similarity_search(query, k=2)
    found_evidence = "\n".join([doc.page_content for doc in results])
    return {"evidence": found_evidence, "schema": get_schema()}


def extract_sql_from_response(text: str) -> str:
    """Extract SQL from LLM response, handling code blocks and plain text."""
    # Try to extract from ```sql ... ``` blocks
    sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try to extract from ``` ... ``` blocks
    code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Look for SELECT/UPDATE/INSERT/DELETE statement in the text
    sql_match = re.search(
        r'(SELECT|UPDATE|INSERT|DELETE)\s+.*?;',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if sql_match:
        return sql_match.group(0).strip()
    
    # Fallback: return the whole text cleaned up
    return text.strip()


def generate_sql_node(state: AgentState):
    llm = build_llm(temperature=0)

    template = """You are a Postgres SQL Agent operating in a PRODUCTION environment.

SCHEMA: {schema}
BUSINESS RULES: {evidence}

CRITICAL INSTRUCTIONS:
1. READ operations (SELECT) are preferred.
2. If the user explicitly asks to UPDATE, CHANGE, or SET data, generate a valid UPDATE statement.
3. ALWAYS include a specific WHERE clause for updates.
4. Do NOT include 'BEGIN' or 'COMMIT'.
5. Return ONLY the raw SQL query inside a ```sql code block.

SPECIAL RULE FOR CHITCHAT/GREETINGS:
If the user input is a greeting (e.g., "hi", "hello", "thanks") or NOT related to the database:
Return exactly: ```sql
SELECT 'Hello! I am your Sales Ops Intelligence Agent. How can I help you with data?' AS greeting;
```

Write a valid SQL query for: {question}"""

    prompt = ChatPromptTemplate.from_template(template)
    inputs = {
        "schema": state['schema'],
        "evidence": state['evidence'],
        "question": state['question'],
    }

    # Preferred path: a validated Pydantic object straight out of the model.
    try:
        result = (prompt | llm.with_structured_output(SQLGeneration)).invoke(inputs)
        if result and result.query.strip():
            return {
                "sql_query": result.query.strip(),
                "sql_reasoning": result.reasoning,
                "model_flagged_write": result.is_write,
                "extraction_mode": "structured",
            }
    except Exception:
        # Small local models frequently emit malformed tool-call JSON.
        # Rather than fail the request, fall through to text extraction.
        pass

    # Fallback path: free-text generation + layered regex extraction.
    response = (prompt | llm).invoke(inputs)
    sql = extract_sql_from_response(response.content)
    return {
        "sql_query": sql,
        "sql_reasoning": "",
        "model_flagged_write": False,
        "extraction_mode": "regex-fallback",
    }


def execute_sql_node(state: AgentState):
    """
    Executes SQL queries with a Strict Human-in-the-Loop for writes.
    - READs (SELECT): Auto-execute immediately.
    - WRITEs (UPDATE/INSERT): Halt, warn, wait for confirmation, then audit & commit.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
      
        raw_query = state['sql_query'].strip()
        clean_query = raw_query.replace("```sql", "").replace("```", "").strip()

        # The generated SQL is right here in front of us, so whether it mutates
        # data is a decidable fact -- not something to ask the model about.
        # Measured: phi3 reports is_write=True for a plain SELECT COUNT(*).
        # So the deterministic scan is authoritative for the gate, and the
        # model's claim is demoted to a cross-check signal we log on mismatch.
        WRITE_KEYWORDS = ("UPDATE ", "INSERT ", "DELETE ", "DROP ", "TRUNCATE ", "ALTER ", "CREATE ")
        is_write_operation = any(cmd in clean_query.upper() for cmd in WRITE_KEYWORDS)

        if state.get("model_flagged_write") != is_write_operation:
            console.print(
                f"[dim]anomaly: model said is_write="
                f"{state.get('model_flagged_write')}, SQL says {is_write_operation}[/dim]"
            )
        
    
        if is_write_operation:
           
            console.print(Panel(
                f"[bold yellow]⚠️  STOP: PRODUCTION WRITE DETECTED[/]\n"
                f"[dim]The Agent is requesting to modify the database:[/dim]\n\n"
                f"[white]{clean_query}[/]\n\n"
                f"[italic red]This action will be logged in 'agent_audit_logs'.[/]",
                title="HUMAN AUTHORIZATION REQUIRED",
                border_style="yellow"
            ))
            
          
            user_approval = Prompt.ask(
                "[bold red]Do you authorize this update?[/]", 
                choices=["yes", "no"], 
                default="no"
            )
            
            if user_approval == "no":
                # Explicitly undo and release. Returning without this leaks a
                # connection holding an open transaction -- and an idle
                # transaction can keep row locks alive in Postgres.
                conn.rollback()
                cur.close()
                conn.close()
                return {
                    "data_result": "❌ ACTION BLOCKED: User denied the update request.",
                    "rows": [],
                    "columns": []
                }
            
            # Hand the reasoning to Postgres as transaction-local context.
            # The AFTER-UPDATE trigger reads it back via current_setting() and
            # writes it into agent_audit_logs alongside the before/after rows.
            # Parameterised via set_config() -- never string-interpolated, or the
            # reasoning text itself becomes an injection vector.
            model_reason = state.get('sql_reasoning') or ''
            reasoning = (
                f"{state.get('question', 'Direct Agent Update')}"
                f"{' | model: ' + model_reason if model_reason else ''}"
            )[:400]
            cur.execute(
                "SELECT set_config('app.current_reason', %s, true);",
                (reasoning,),
            )

            cur.execute(clean_query)
            affected = cur.rowcount
            
            
            conn.commit()
            
            result_str = (
                f"✅ SUCCESS: {affected} row(s) modified.\n"
                f"🔒 AUDIT: Change logged securely."
            )
            
            cur.close()
            conn.close()
            return {"data_result": result_str, "rows": [], "columns": []}

   
        else:
            cur.execute(clean_query)
            
            if cur.description:
                rows = cur.fetchall()
                colnames = [desc[0] for desc in cur.description]
                # Format for AI consumption
                data_preview = [dict(zip(colnames, row)) for row in rows]
                result_str = f"PREVIEW SUCCESS: Found {len(rows)} record(s).\nData: {data_preview}"
            else:
                rows, colnames = [], []
                result_str = "Query executed successfully. No data returned."
                
            cur.close()
            conn.close()
            return {"data_result": result_str, "rows": rows, "columns": colnames}

    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        return {"data_result": f"CRITICAL SQL ERROR: {str(e)}", "rows": [], "columns": []}


def answer_node(state: AgentState):
    llm = build_llm(temperature=0)
    response = llm.invoke(f"Question: {state['question']}\nData: {state['data_result']}\nAnswer professionally in 2-3 sentences:")
    return {"final_answer": response.content}


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

def print_header():
    console.clear()
    console.print(Panel.fit(
        "[bold white]SALES OPS INTELLIGENCE AGENT[/bold white]\n"
        "[dim]v2.2 | Local LLM Mode (Ollama + Phi3) | Unlimited[/dim]",
        style="bold blue",
        subtitle="Type 'exit' to quit"
    ))

def print_sql_panel(sql):
    syntax = Syntax(sql, "sql", theme="monokai", line_numbers=False)
    console.print(Panel(syntax, title="[bold yellow]Generated SQL[/]", border_style="yellow"))

def print_data_table(columns, rows):
    if not rows:
        return
    
    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    
    for col in columns:
        table.add_column(col)
    
    for row in rows[:10]:
        table.add_row(*[str(cell) for cell in row])
        
    if len(rows) > 10:
        console.print(f"[dim]Showing top 10 of {len(rows)} rows...[/dim]")
    
    console.print(table)


if __name__ == "__main__":
    print_header()
    
    while True:
        try:
            console.print()
            user_q = Prompt.ask("[bold green]Ask a question[/]")
            
            if user_q.lower() in ["exit", "quit", "q"]:
                console.print("[bold red]Shutting down system...[/]")
                break
            
            console.print("[bold cyan]Analyzing request...[/]")
            state = app.invoke({"question": user_q})
                
           
            if "UPDATE" not in state['sql_query'].upper():
                print_sql_panel(state['sql_query'])
            
            if "SQL ERROR" in state['data_result']:
                console.print(f"[bold red] {state['data_result']}[/]")
            elif "ACTION BLOCKED" in state['data_result']:
                console.print(f"[bold yellow] {state['data_result']}[/]")
            else:
                #
                if state.get('rows'):
                    print_data_table(state.get('columns', []), state.get('rows', []))
                else:
                    console.print(f"[dim]{state['data_result']}[/dim]")
            
            console.print(Panel(Markdown(state['final_answer']), title="[bold green]Insight[/]", border_style="green"))

        except Exception as e:
            console.print(f"[bold red]System Error: {str(e)}[/]")