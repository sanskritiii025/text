import os
from dotenv import load_dotenv
load_dotenv()
"""Quick diagnostic script to find what's broken."""
import traceback

# 1. Test DB connection
print("=" * 50)
print("TEST 1: Database Connection")
print("=" * 50)
try:
    import psycopg2
    conn = psycopg2.connect(
        dbname="postgres", user="postgres",
        password=os.getenv("DB_PASSWORD"), host="localhost", port="5432"
    )
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = cur.fetchall()
    print(f"  OK - Found {len(tables)} tables: {[t[0] for t in tables]}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"  FAIL: {e}")

# 2. Test ChromaDB / vector store
print("\n" + "=" * 50)
print("TEST 2: ChromaDB Vector Store")
print("=" * 50)
try:
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    vector_db = Chroma(
        persist_directory="./sales_memory",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    )
    results = vector_db.similarity_search("high value deals", k=2)
    print(f"  OK - Retrieved {len(results)} docs")
    for i, doc in enumerate(results):
        print(f"  Doc {i+1}: {doc.page_content[:100]}...")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# 3. Test Gemini LLM
print("\n" + "=" * 50)
print("TEST 3: Gemini LLM (structured output)")
print("=" * 50)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import BaseModel, Field

    class SQLOutput(BaseModel):
        reasoning: str = Field(description="Why you wrote this query")
        sql: str = Field(description="The PostgreSQL query")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    structured_llm = llm.with_structured_output(SQLOutput)
    resp = structured_llm.invoke("Write a SQL query to count rows in a table called 'opportunities'")
    print(f"  OK - Got SQL: {resp.sql}")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# 4. Test full agent pipeline
print("\n" + "=" * 50)
print("TEST 4: Full Agent Pipeline")
print("=" * 50)
try:
    from sales_agent import app
    state = app.invoke({"question": "How many opportunities are there?"})
    print(f"  SQL: {state.get('sql_query', 'N/A')}")
    print(f"  Data: {state.get('data_result', 'N/A')[:200]}")
    print(f"  Answer: {state.get('final_answer', 'N/A')[:200]}")
    print("  OK - Full pipeline works!")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

print("\n" + "=" * 50)
print("DIAGNOSTICS COMPLETE")
print("=" * 50)
