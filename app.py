import os, re, json, sqlite3
import google.generativeai as genai
from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ------------------ CONFIG ------------------
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# ------------------ STATE ------------------
class TicketState(TypedDict, total=False):
    subject: str
    description: str
    category: str
    retrieved_context: list
    draft_response: str
    review_result: dict
    attempts: int
    escalated: bool
    escalation_message: str

# ------------------ NODES (classify, retrieve, draft, review, retry, escalate) ------------------
# … (reuse your node functions here, same as we built earlier) …

# ------------------ BUILD GRAPH ------------------
builder = StateGraph(TicketState)

builder.add_node("classify", classification_node)
builder.add_node("retrieve", retrieval_node)
builder.add_node("draft", draft_node)
builder.add_node("review", reviewer_node)
builder.add_node("retry", retry_node)
builder.add_node("escalate", escalation_node)

builder.set_entry_point("classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "draft")
builder.add_edge("draft", "review")

def review_cond(state: TicketState):
    return "approved" if state["review_result"]["approved"] else "rejected"
builder.add_conditional_edges("review", review_cond,
    {"approved": END, "rejected": "retry"})

def retry_cond(state: TicketState):
    return "fail" if state.get("attempts", 0) >= 2 else "retry"
builder.add_edge("retry", "retrieve")
builder.add_conditional_edges("retry", retry_cond,
    {"retry": "retrieve", "fail": "escalate"})
builder.add_edge("escalate", END)

# persistent SQLite checkpointer
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = builder.compile(checkpointer=checkpointer)

# expose graph for CLI
app = graph
