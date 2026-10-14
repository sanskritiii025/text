"""Rx Commercial Insights: ask Medicare Part D prescribing data questions in English.

Run locally:  streamlit run app.py
"""
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent.db import PostgresDB            # noqa: E402  (env must be loaded first)
from agent.graph import build_agent        # noqa: E402
from agent.knowledge import Glossary       # noqa: E402
from agent.llm import model_label          # noqa: E402

# Demo sign-in. In production each user would authenticate (SSO) and be mapped
# to their database role; here you pick the role to see RLS in action.
PERSONAS = {
    "National Sales Director (all territories)": "rx_national",
    "Territory Manager - Hartford": "mgr_hartford",
    "Territory Manager - New Haven": "mgr_new_haven",
    "Territory Manager - Stamford": "mgr_stamford",
    "Territory Manager - Danbury": "mgr_danbury",
    "Territory Manager - Waterbury": "mgr_waterbury",
    "Territory Manager - Norwich": "mgr_norwich",
}
EXAMPLES = [
    "What was Eliquis's market share in 2023?",
    "How did Jardiance and Farxiga claims trend from 2021 to 2023?",
    "Top 10 GLP-1 prescribers in 2023, with specialty and city",
    "Which cardiologists write the most anticoagulant claims but give Eliquis less than half of them?",
    "Which specialties drive Ozempic volume?",
    "Top 5 brands by total drug cost in 2023 and their cost per claim",
]
QUESTION_LIMIT = int(os.getenv("SESSION_QUESTION_LIMIT", "25"))   # protects the free-tier LLM quota

st.set_page_config(page_title="Rx Commercial Insights", page_icon="💊", layout="wide")


@st.cache_resource(show_spinner=False)
def get_glossary():
    return Glossary()


@st.cache_resource(show_spinner=False)
def get_agent(role: str):
    db = PostgresDB(role)
    return db, build_agent(db, get_glossary())


@st.cache_data(ttl=600, show_spinner=False)
def scope_summary(role: str):
    db, _ = get_agent(role)
    r = db.run("SELECT (SELECT count(*) FROM prescribers), "
               "(SELECT string_agg(t.territory_name, ', ' ORDER BY t.territory_name) "
               " FROM territories t JOIN user_territories u USING (territory_id))")
    return r.rows[0]


def auto_chart(df: pd.DataFrame):
    """Pick a sensible chart from the shape of the result, or none."""
    if df.empty or len(df) < 2:
        return
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "year"]
    text = [c for c in df.columns if df[c].dtype == object]
    if not numeric:
        return
    if "year" in df.columns:
        if text and df[text[0]].nunique() <= 8:
            wide = df.pivot_table(index="year", columns=text[0], values=numeric[0], aggfunc="sum")
            st.line_chart(wide)
        elif not text:
            st.line_chart(df.set_index("year")[numeric[0]])
    elif text and len(df) <= 30:
        st.bar_chart(df.set_index(text[0])[numeric[0]], horizontal=True)


def to_frame(result) -> pd.DataFrame:
    df = pd.DataFrame(result.rows, columns=result.columns)
    for c in df.columns:   # psycopg2 returns Decimal; charts need floats
        if df[c].map(lambda v: v.__class__.__name__ == "Decimal").any():
            df[c] = df[c].astype(float)
    return df


def render_turn(turn):
    st.markdown(turn["answer"])
    if turn.get("df") is not None and not turn["df"].empty:
        auto_chart(turn["df"])
        st.dataframe(turn["df"], width="stretch", hide_index=True)
        st.download_button("Download CSV", turn["df"].to_csv(index=False), "result.csv",
                           key=f"dl{turn['id']}")
    with st.expander("How I got this"):
        retries = len(turn["attempts"]) - 1
        st.caption(f"{turn['elapsed']:.1f}s · {len(turn['attempts'])} attempt(s)"
                   + (f" · self-corrected {retries}x" if retries and not turn["failed"] else ""))
        if turn.get("reasoning"):
            st.markdown(f"**Plan:** {turn['reasoning']}")
        st.code(turn["sql"], language="sql")
        for i, a in enumerate(turn["attempts"][:-1] if not turn["failed"] else turn["attempts"], 1):
            st.markdown(f"**Attempt {i} failed:** `{a['error']}`")
            st.code(a["sql"], language="sql")
        st.markdown("**Definitions used:**")
        for e in turn["evidence"]:
            st.markdown(f"- {e.splitlines()[0].removeprefix('## ')}")


# ── Sidebar ──
with st.sidebar:
    st.header("💊 Rx Commercial Insights")
    persona = st.selectbox("Signed in as", list(PERSONAS))
    role = PERSONAS[persona]
    if st.session_state.get("role") != role:
        st.session_state.update(role=role, turns=[], asked=0)
    try:
        n_hcps, territories = scope_summary(role)
        st.success(f"Row-level security: you can see **{n_hcps:,} prescribers** in {territories}.")
    except Exception as e:
        st.error(f"Database unavailable: {e}")
        st.stop()
    st.caption(f"Data: CMS Medicare Part D, Connecticut, 2021-2023 · Model: {model_label()}")
    counter = st.empty()   # filled at the end of the run, after this turn's question is counted
    st.markdown("**Try asking**")
    for q in EXAMPLES:
        if st.button(q, width="stretch"):
            st.session_state.pending = q
    if st.button("Clear conversation", type="secondary"):
        st.session_state.update(turns=[])

# ── Conversation ──
for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_turn(turn)

question = st.chat_input("Ask about prescribers, brands, markets, territories...") or st.session_state.pop("pending", None)
if question:
    if st.session_state.asked >= QUESTION_LIMIT:
        st.warning("Question limit reached for this session (the demo runs on a free LLM tier).")
        st.stop()
    st.session_state.asked += 1
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        _, agent = get_agent(role)
        history = [(t["question"], t["sql"]) for t in st.session_state.turns if not t["failed"]]
        start = time.time()
        with st.spinner("Retrieving definitions, writing SQL, querying..."):
            try:
                state = agent.invoke({"question": question, "history": history})
            except Exception as e:
                state = {"attempts": [{"sql": "", "error": str(e)}], "error": str(e), "evidence": []}
        failed = bool(state.get("error")) or state.get("result") is None
        turn = {
            "id": len(st.session_state.turns),
            "question": question,
            "answer": state.get("answer") or f"I couldn't answer that: `{state.get('error')}`. Try rephrasing.",
            "df": None if failed else to_frame(state["result"]),
            "sql": state.get("sql", ""),
            "reasoning": state.get("reasoning", ""),
            "attempts": state.get("attempts", []),
            "evidence": state.get("evidence", []),
            "elapsed": time.time() - start,
            "failed": failed,
        }
        render_turn(turn)
    st.session_state.turns.append(turn)

counter.caption(f"Questions this session: {st.session_state.asked}/{QUESTION_LIMIT}")
