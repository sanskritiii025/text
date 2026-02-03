import os
import time
import psycopg2
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.status import Status

console = Console()

DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "Sanskriti@25",  
    "host": "localhost",
    "port": "5432"
}

vector_db = Chroma(
    persist_directory="./sales_memory",
    embedding_function=OpenAIEmbeddings()
)

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


class SQLOutput(BaseModel):
    reasoning: str = Field(description="Why you wrote this query")
    sql: str = Field(description="The PostgreSQL query")

def generate_sql_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    structured_llm = llm.with_structured_output(SQLOutput)
    
    template = """
    You are a Postgres SQL Agent operating in a PRODUCTION environment.
    
    SCHEMA: {schema}
    BUSINESS RULES: {evidence}
    
    
    CRITICAL INSTRUCTIONS:
    1. READ operations (SELECT) are preferred.
    2. If the user explicitly asks to UPDATE, CHANGE, or SET data, generate a valid UPDATE statement.
    3. ALWAYS include a specific WHERE clause for updates.
    4. Do NOT include 'BEGIN' or 'COMMIT'.
    5. Return ONLY the raw SQL query.
    
    SPECIAL RULE FOR CHITCHAT/GREETINGS:
    If the user input is a greeting (e.g., "hi", "hello", "thanks") or NOT related to the database:
    Return exactly: SELECT 'Hello! I am your Sales Ops Intelligence Agent. How can I help you with data?' AS greeting;
    
    Write a valid SQL query for: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    response = (prompt | structured_llm).invoke({
        "schema": state['schema'],
        "evidence": state['evidence'],
        "question": state['question']
    })
    
    return {"sql_query": response.sql}


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
        
      
        is_write_operation = any(cmd in clean_query.upper() for cmd in ["UPDATE ", "INSERT ", "DELETE "])
        
    
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
                return {
                    "data_result": "❌ ACTION BLOCKED: User denied the update request.", 
                    "rows": [], 
                    "columns": []
                }
            
            #
            reasoning = state.get('question', 'Direct Agent Update')[:200].replace("'", "")
            cur.execute(f"SET LOCAL app.current_reason = '{reasoning}';")
            
            
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
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    response = llm.invoke(f"Question: {state['question']}\nData: {state['data_result']}\nAnswer professionally:")
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
        "[dim]v2.1 | Auto-Read / Gated-Write Mode[/dim]",
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