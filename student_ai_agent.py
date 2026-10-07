"""
Student Records Agent - LangChain + LangGraph + Groq (free tier) + SQLite

Setup:
    pip install -U langchain-groq langchain-core langgraph
    export GROQ_API_KEY="gsk_..."        # free key from https://console.groq.com/keys
    python student_agent.py

Everything is free/local: Groq free-tier LLM + a local SQLite file. No paid API.
The LLM never writes raw SQL; it only calls the safe, parameterized tools below.
"""

import json
import os
import sqlite3

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

DB_PATH = "students.db"
# Groq retires models often. llama-3.3-70b-versatile was shut down on 2026-08-16.
# Override without editing code:  export GROQ_MODEL="openai/gpt-oss-20b"
# Other options: "qwen/qwen3.6-27b". Check https://console.groq.com/docs/models
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ----------------------------------------------------------------------------
# Database layer
# ----------------------------------------------------------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                roll_no    TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                department TEXT NOT NULL,
                year       INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS marks (
                roll_no TEXT NOT NULL REFERENCES students(roll_no),
                subject TEXT NOT NULL,
                marks   REAL NOT NULL CHECK (marks BETWEEN 0 AND 100),
                PRIMARY KEY (roll_no, subject)
            );
            """
        )
        if conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO students VALUES (?,?,?,?)",
                [
                    ("S001", "Aarav Sharma", "CSE", 2),
                    ("S002", "Diya Reddy", "ECE", 3),
                    ("S003", "Rohan Mehta", "CSE", 2),
                    ("S004", "Ananya Iyer", "EEE", 1),
                    ("S005", "Kabir Singh", "ECE", 3),
                ],
            )
            conn.executemany(
                "INSERT INTO marks VALUES (?,?,?)",
                [
                    ("S001", "Maths", 88), ("S001", "Physics", 76), ("S001", "Programming", 94),
                    ("S002", "Maths", 72), ("S002", "Signals", 85), ("S002", "Electronics", 91),
                    ("S003", "Maths", 65), ("S003", "Physics", 58), ("S003", "Programming", 80),
                    ("S004", "Maths", 90), ("S004", "Physics", 87),
                    ("S005", "Maths", 55), ("S005", "Signals", 62), ("S005", "Electronics", 70),
                ],
            )


# ----------------------------------------------------------------------------
# Tools (the agent's only way to touch the data)
# ----------------------------------------------------------------------------
@tool
def list_students(department: str = "", year: int = 0) -> str:
    """List students. Optionally filter by department (e.g. 'CSE') and/or year (1-4).
    Leave department empty and year 0 for no filter."""
    q, params = "SELECT * FROM students WHERE 1=1", []
    if department:
        q += " AND UPPER(department)=UPPER(?)"
        params.append(department)
    if year:
        q += " AND year=?"
        params.append(year)
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(q + " ORDER BY roll_no", params)]
    return json.dumps(rows) if rows else "No students found."


@tool
def search_student_by_name(name: str) -> str:
    """Find students whose name contains the given text (case-insensitive)."""
    with get_conn() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM students WHERE name LIKE ?", (f"%{name}%",)
            )
        ]
    return json.dumps(rows) if rows else f"No student matching '{name}'."


@tool
def get_student_report(roll_no: str) -> str:
    """Get a student's profile, all subject marks, total, average and grade by roll number."""
    with get_conn() as conn:
        s = conn.execute(
            "SELECT * FROM students WHERE roll_no=?", (roll_no.upper(),)
        ).fetchone()
        if not s:
            return f"No student with roll number {roll_no}."
        marks = [
            dict(r)
            for r in conn.execute(
                "SELECT subject, marks FROM marks WHERE roll_no=?", (roll_no.upper(),)
            )
        ]
    avg = round(sum(m["marks"] for m in marks) / len(marks), 2) if marks else None
    grade = None
    if avg is not None:
        grade = "A" if avg >= 85 else "B" if avg >= 70 else "C" if avg >= 55 else "D" if avg >= 40 else "F"
    return json.dumps({"student": dict(s), "marks": marks, "average": avg, "grade": grade})


@tool
def add_student(roll_no: str, name: str, department: str, year: int) -> str:
    """Add a new student record."""
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO students VALUES (?,?,?,?)",
                (roll_no.upper(), name, department.upper(), year),
            )
        return f"Added {name} ({roll_no.upper()})."
    except sqlite3.IntegrityError:
        return f"Roll number {roll_no} already exists."


@tool
def set_marks(roll_no: str, subject: str, marks: float) -> str:
    """Add or update a student's marks (0-100) for a subject."""
    if not 0 <= marks <= 100:
        return "Marks must be between 0 and 100."
    with get_conn() as conn:
        if not conn.execute(
            "SELECT 1 FROM students WHERE roll_no=?", (roll_no.upper(),)
        ).fetchone():
            return f"No student with roll number {roll_no}."
        conn.execute(
            "INSERT INTO marks VALUES (?,?,?) "
            "ON CONFLICT(roll_no, subject) DO UPDATE SET marks=excluded.marks",
            (roll_no.upper(), subject.title(), marks),
        )
    return f"Set {subject.title()} = {marks} for {roll_no.upper()}."


@tool
def top_students(n: int = 3, department: str = "") -> str:
    """Rank students by average marks (highest first). Optional department filter."""
    q = (
        "SELECT s.roll_no, s.name, s.department, ROUND(AVG(m.marks),2) AS average "
        "FROM students s JOIN marks m ON m.roll_no=s.roll_no "
    )
    params: list = []
    if department:
        q += "WHERE UPPER(s.department)=UPPER(?) "
        params.append(department)
    q += "GROUP BY s.roll_no ORDER BY average DESC LIMIT ?"
    params.append(max(1, min(n, 50)))
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(q, params)]
    return json.dumps(rows) if rows else "No data."


@tool
def subject_stats(subject: str) -> str:
    """Class statistics (count, average, highest, lowest) for one subject."""
    with get_conn() as conn:
        r = conn.execute(
            "SELECT COUNT(*) c, ROUND(AVG(marks),2) avg, MAX(marks) hi, MIN(marks) lo "
            "FROM marks WHERE LOWER(subject)=LOWER(?)",
            (subject,),
        ).fetchone()
    if not r["c"]:
        return f"No marks recorded for {subject}."
    return json.dumps(dict(r))


TOOLS = [
    list_students, search_student_by_name, get_student_report,
    add_student, set_marks, top_students, subject_stats,
]

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant for managing student records. "
        "Always use the provided tools to read or change data; never invent records. "
        "If a student is referred to by name, search by name first to get the roll number. "
        "Be concise and present results clearly."
    )
)


# ----------------------------------------------------------------------------
# LangGraph: agent <-> tools loop
# ----------------------------------------------------------------------------
def build_graph():
    llm = ChatGroq(model=MODEL, temperature=0)  # reads GROQ_API_KEY from env
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: MessagesState):
        response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    # MemorySaver keeps conversation history per thread_id (in-memory, free)
    return graph.compile(checkpointer=MemorySaver())


import getpass

def main():
    if not os.getenv(""):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Paste your Groq API key: ")   


    init_db()
    app = build_graph()
    config = {"configurable": {"thread_id": "student-session-1"}}

    print("Student Records Agent ready. Type 'exit' to quit.")
    print("Try: 'Show CSE students', 'Report for Diya Reddy', 'Top 3 students', "
          "'Set Rohan's Physics marks to 72'\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        try:
            result = app.invoke({"messages": [HumanMessage(content=user)]}, config)
            print(f"Agent: {result['messages'][-1].content}\n")
        except Exception as e:  # e.g. Groq free-tier rate limit
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
