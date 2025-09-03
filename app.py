import os, re, json, pandas as pd, streamlit as st
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List, Dict

# ------------------ CONFIG ------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ Please add GEMINI_API_KEY in Streamlit secrets!")
else:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ------------------ STATE ------------------
class TicketState(TypedDict, total=False):
    subject: str
    description: str
    category: str
    classification_explanation: str
    retrieved_context: list
    draft_response: str
    review_result: dict
    attempts: int
    escalated: bool
    escalation_message: str

# ------------------ MOCK KB ------------------
KNOWLEDGE_BASE = {
    "Billing": ["Refunds require 5–7 business days.", "Invoices auto-send monthly."],
    "Technical": ["Clear cache if app crashes.", "API supports REST + GraphQL."],
    "Security": ["Enable 2FA.", "Report phishing to security@company.com."],
    "General": ["Support is 24/7 via chat.", "Reports export as PDF/CSV."],
}

# ------------------ CLASSIFIER ------------------
CATEGORIES = ["Billing", "Technical", "Security", "General"]
KEYWORDS: Dict[str, List[str]] = {
    "Billing": ["refund", "invoice", "payment", "charged", "subscription"],
    "Technical": ["bug", "error", "crash", "login", "api"],
    "Security": ["hacked", "phish", "security", "2fa"],
}

def classification_node(state: TicketState) -> TicketState:
    text = (state["subject"] + " " + state["description"]).lower()
    scores = {c: 0 for c in CATEGORIES}
    for cat, keys in KEYWORDS.items():
        for k in keys:
            if k in text:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return {**state, "category": best if scores[best] > 0 else "General"}

# ------------------ RETRIEVAL ------------------
def retrieval_node(state: TicketState) -> TicketState:
    ctx = KNOWLEDGE_BASE.get(state["category"], [])[:2]
    return {**state, "retrieved_context": ctx}

# ------------------ GEMINI DRAFT ------------------
draft_model = genai.GenerativeModel("gemini-1.5-flash")
def draft_node(state: TicketState) -> TicketState:
    ctx = "\n".join(state.get("retrieved_context", []))
    prompt = f"""
You are a support assistant.

Ticket:
{state['subject']} — {state['description']}
Category: {state['category']}
Context: {ctx}

Reply politely, professionally.
Do NOT promise refunds directly.
End by offering further help.
"""
    resp = draft_model.generate_content(prompt)
    return {**state, "draft_response": resp.text.strip()}

# ------------------ GEMINI REVIEW ------------------
review_model = genai.GenerativeModel("gemini-1.5-flash")
def reviewer_node(state: TicketState) -> TicketState:
    prompt = f"""
Review this draft for compliance.

Ticket: {state['subject']} — {state['description']}
Draft: {state['draft_response']}

Rules:
- Must be polite
- No direct refund promises
- No unsafe security advice

Return JSON:
{{"approved": true/false, "feedback": "..." }}
"""
    resp = review_model.generate_content(prompt).text.strip()
    try:
        match = re.search(r"\{.*\}", resp, re.S)
        result = json.loads(match.group(0))
    except:
        result = {"approved": False, "feedback": f"Parse error: {resp}"}
    return {**state, "review_result": result}

# ------------------ RETRY + ESCALATION ------------------
def retry_node(state: TicketState) -> TicketState:
    return {**state, "attempts": state.get("attempts", 0) + 1}

def escalation_node(state: TicketState) -> TicketState:
    return {**state, "escalated": True, "escalation_message": "Escalated to human after 2 fails."}

# ------------------ BUILD GRAPH ------------------
checkpointer = SqliteSaver.from_conn_string(":memory:")  # in-memory SQLite
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
builder.add_conditional_edges("review", review_cond, {"approved": END, "rejected": "retry"})

def retry_cond(state: TicketState):
    return "fail" if state.get("attempts", 0) >= 2 else "retry"
builder.add_edge("retry", "retrieve")
builder.add_conditional_edges("retry", retry_cond, {"retry": "retrieve", "fail": "escalate"})
builder.add_edge("escalate", END)

graph = builder.compile(checkpointer=checkpointer)

# ------------------ STREAMLIT UI ------------------
st.title("🧾 Support Ticket Resolution Agent")

with st.form("ticket_form"):
    subject = st.text_input("Ticket Subject")
    description = st.text_area("Ticket Description")
    submitted = st.form_submit_button("Submit")

if submitted and subject and description:
    with st.spinner("Processing ticket..."):
        result = graph.invoke(
            {"subject": subject, "description": description},
            config={"configurable": {"thread_id": f"ticket-{hash(subject+description)}"}}
        )

    st.subheader("Results")
    st.write("**Category:**", result.get("category"))
    st.write("**Context:**", result.get("retrieved_context"))
    st.write("**Draft:**")
    st.info(result.get("draft_response"))
    st.write("**Review:**", result.get("review_result"))

    if result.get("escalated"):
        st.error(result.get("escalation_message"))
