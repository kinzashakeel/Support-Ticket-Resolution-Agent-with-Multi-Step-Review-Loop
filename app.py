from typing import TypedDict, Optional, Dict, List
import re

CATEGORIES = ["Billing", "Technical", "Security", "General"]

# Toggle: keep this False for fast, dependency-free classification.
USE_ZEROSHOT = False  # set True later if you want ML-based classification

# Simple keyword map for fast classification
KEYWORDS: Dict[str, List[str]] = {
    "Billing": [
        r"refund", r"invoice", r"payment", r"charged", r"billing", r"subscription", r"credit card", r"price"
    ],
    "Technical": [
        r"bug", r"error", r"issue", r"crash", r"not loading", r"api", r"integration",
        r"install", r"login", r"authentication", r"performance", r"timeout", r"mobile", r"web"
    ],
    "Security": [
        r"breach", r"hacked", r"compromise", r"phish", r"phishing", r"2fa", r"mfa", r"security",
        r"unauthorized", r"privacy", r"data leak"
    ],
}

def classify_rule_based(subject: str, description: str):
    text = f"{subject}\n{description}".lower()
    scores = {cat: 0 for cat in CATEGORIES}
    hits = {cat: [] for cat in CATEGORIES}

    for cat, patterns in KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text):
                scores[cat] += 1
                hits[cat].append(pat)

    # pick highest; default to General
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "General"

    explanation = f"Rule-based classification → {best}. Keyword hits: " + \
                  ", ".join(f"{c}:{len(h)}" for c,h in hits.items())
    return best, explanation


from langgraph.graph import StateGraph, END

class TicketState(TypedDict, total=False):
    # input
    subject: str
    description: str
    # outputs from this module
    category: str
    classification_explanation: str

def classification_node(state: TicketState) -> TicketState:
    subject = state.get("subject", "")
    description = state.get("description", "")

    if USE_ZEROSHOT:
        category, explanation = classify_zeroshot(subject, description)
    else:
        category, explanation = classify_rule_based(subject, description)

    return {
        **state,
        "category": category,
        "classification_explanation": explanation,
    }
# Build graph with one node
builder = StateGraph(TicketState)
builder.add_node("classify", classification_node)
builder.set_entry_point("classify")
builder.add_edge("classify", END)
graph = builder.compile()

# Some sample tickets to test
samples = [
    {
        "subject": "Charged twice this month",
        "description": "My card shows two payments for the same subscription."
    },
    {
        "subject": "Mobile app crashes on login",
        "description": "Android app closes whenever I try to sign in via Google."
    },
    {
        "subject": "Suspicious email about password reset",
        "description": "I received a phishing-looking email asking for my credentials."
    },
    {
        "subject": "Question about new features",
        "description": "Do you support exporting reports as CSV?"
    },
]

for i, s in enumerate(samples, 1):
    result = graph.invoke(s)
    print(f"--- Sample {i} ---")
    print("Subject:", s["subject"])
    print("Category:", result["category"])
    print("Explanation:", result["classification_explanation"])
    print()
# Simple mock KBs for each category
KNOWLEDGE_BASE = {
    "Billing": [
        "Refunds require 5–7 business days to process.",
        "Invoices are sent automatically at the start of each billing cycle.",
        "Payment failures may occur if the credit card is expired."
    ],
    "Technical": [
        "If the app crashes, try reinstalling and clearing cache.",
        "Login issues may be resolved by resetting your password.",
        "Our API supports REST and GraphQL endpoints."
    ],
    "Security": [
        "Always enable 2FA for better protection.",
        "We never ask for your password over email.",
        "Report phishing attempts to security@company.com."
    ],
    "General": [
        "You can reach support 24/7 via email or chat.",
        "New features are announced in our monthly newsletter.",
        "Exporting reports is supported in PDF and CSV formats."
    ],
}
def retrieve_context(subject: str, description: str, category: str, top_k: int = 2):
    """
    Simple retrieval: return top_k items from category KB that
    contain any keyword match, else fallback to first items.
    """
    text = f"{subject} {description}".lower()
    docs = KNOWLEDGE_BASE.get(category, [])

    # prioritize matches
    matched = [d for d in docs if any(word in text for word in d.lower().split())]
    if not matched:
        matched = docs[:top_k]  # fallback

    return matched[:top_k]
class TicketState(TypedDict, total=False):
    # existing
    subject: str
    description: str
    category: str
    classification_explanation: str
    # new
    retrieved_context: list

def retrieval_node(state: TicketState) -> TicketState:
    subject = state["subject"]
    description = state["description"]
    category = state["category"]

    context = retrieve_context(subject, description, category)
    return {
        **state,
        "retrieved_context": context
    }
# new graph
builder = StateGraph(TicketState)
builder.add_node("classify", classification_node)
builder.add_node("retrieve", retrieval_node)

builder.set_entry_point("classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", END)

graph = builder.compile()

# test again
samples = [
    {
        "subject": "Charged twice this month",
        "description": "My card shows two payments for the same subscription."
    },
    {
        "subject": "Mobile app crashes on login",
        "description": "Android app closes whenever I try to sign in via Google."
    },
    {
        "subject": "Suspicious email about password reset",
        "description": "I received a phishing-looking email asking for my credentials."
    },
]

for i, s in enumerate(samples, 1):
    result = graph.invoke(s)
    print(f"--- Sample {i} ---")
    print("Category:", result["category"])
    print("Retrieved context:", result["retrieved_context"])
    print()
import google.generativeai as genai
import os

os.environ["GEMINI_API_KEY"] = "AIzaSyAKIq-OXzVZC7lTYtnZMdDFqiHDQ_a3L3o"  # 🔑 put your key here
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

draft_model = genai.GenerativeModel("gemini-1.5-flash")

def generate_draft_with_gemini(subject: str, description: str, category: str, context: list):
    context_text = "\n".join([f"- {c}" for c in context]) if context else "No extra context available."

    prompt = f"""
You are a professional customer support assistant.

Ticket:
Subject: {subject}
Description: {description}

Category: {category}

Relevant Context:
{context_text}

Write a clear, polite, and helpful draft response for the customer.
- Keep the tone friendly and professional.
- Do NOT promise refunds or unsafe security actions directly.
- End with an offer of further assistance.
    """

    response = draft_model.generate_content(prompt)
    return response.text.strip()
review_model = genai.GenerativeModel("gemini-1.5-flash")

def review_draft(ticket, draft):
    prompt = f"""
You are a reviewer for a customer support agent.
Check if the draft reply is polite, safe, and compliant.

Ticket:
Subject: {ticket['subject']}
Description: {ticket['description']}

Draft Response:
{draft}

Review Rules:
- Must be polite/helpful
- No direct refund promises (just say "we will check with billing team")
- No unsafe security advice
- Must sound professional

Reply ONLY in JSON with this schema:
{{
  "approved": true/false,
  "feedback": "short feedback"
}}
    """

    response = review_model.generate_content(prompt)
    text = response.text.strip()

    import json, re
    try:
        match = re.search(r"\{.*\}", text, re.S)
        result = json.loads(match.group(0))
    except:
        result = {"approved": False, "feedback": f"Could not parse reviewer output: {text}"}
    return result
from typing import TypedDict
from langgraph.graph import StateGraph, END

class TicketState(TypedDict, total=False):
    subject: str
    description: str
    category: str
    classification_explanation: str
    retrieved_context: list
    draft_response: str
    review_result: dict

# Draft Node
def draft_node(state: TicketState) -> TicketState:
    draft = generate_draft_with_gemini(
        state["subject"],
        state["description"],
        state["category"],
        state.get("retrieved_context", [])
    )
    return {**state, "draft_response": draft}

# Reviewer Node
def reviewer_node(state: TicketState) -> TicketState:
    ticket = {"subject": state["subject"], "description": state["description"]}
    result = review_draft(ticket, state["draft_response"])
    return {**state, "review_result": result}
builder = StateGraph(TicketState)
builder.add_node("classify", classification_node)   # from Module 1
builder.add_node("retrieve", retrieval_node)        # from Module 2
builder.add_node("draft", draft_node)               # Gemini draft
builder.add_node("review", reviewer_node)           # Gemini review

builder.set_entry_point("classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "draft")
builder.add_edge("draft", "review")
builder.add_edge("review", END)

graph = builder.compile()
samples = [
    {
        "subject": "Charged twice this month",
        "description": "My card shows two payments for the same subscription."
    },
    {
        "subject": "Mobile app crashes on login",
        "description": "Android app closes whenever I try to sign in via Google."
    }
]

for i, s in enumerate(samples, 1):
    result = graph.invoke(s)
    print(f"--- Ticket {i} ---")
    print("Category:", result["category"])
    print("Draft Response:\n", result["draft_response"])
    print("Review Result:", result["review_result"])
    print()
class TicketState(TypedDict, total=False):
    subject: str
    description: str
    category: str
    classification_explanation: str
    retrieved_context: list
    draft_response: str
    review_result: dict
    attempts: int
def reviewer_node(state: TicketState) -> TicketState:
    ticket = {"subject": state["subject"], "description": state["description"]}
    result = review_draft(ticket, state["draft_response"])
    return {**state, "review_result": result}
def retry_node(state: TicketState) -> TicketState:
    attempts = state.get("attempts", 0) + 1
    return {**state, "attempts": attempts}
def escalation_node(state: TicketState) -> TicketState:
    msg = f"Escalated after {state.get('attempts', 0)} failed attempts."
    return {**state, "escalated": True, "escalation_message": msg}
builder = StateGraph(TicketState)

builder.add_node("classify", classification_node)
builder.add_node("retrieve", retrieval_node)
builder.add_node("draft", draft_node)
builder.add_node("review", reviewer_node)
builder.add_node("retry", retry_node)
builder.add_node("escalate", escalation_node)

builder.set_entry_point("classify")

# main flow
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "draft")
builder.add_edge("draft", "review")

# conditional branching from review
def review_condition(state: TicketState):
    if state["review_result"]["approved"]:
        return "approved"
    return "rejected"

builder.add_conditional_edges(
    "review",
    review_condition,
    {"approved": END, "rejected": "retry"}
)

# retry flow
def retry_condition(state: TicketState):
    if state.get("attempts", 0) >= 2:
        return "fail"
    return "retry"

builder.add_edge("retry", "retrieve")  # re-run retrieval & draft
builder.add_conditional_edges(
    "retry",
    retry_condition,
    {"retry": "retrieve", "fail": "escalate"}
)

builder.add_edge("escalate", END)

graph = builder.compile()
# force a problematic draft to test retry
sample = {
    "subject": "Refund request",
    "description": "I demand an immediate refund for my last payment."
}

result = graph.invoke(sample)
print("Category:", result["category"])
print("Draft:", result["draft_response"][:200], "...")
print("Review:", result["review_result"])
print("Attempts:", result.get("attempts"))
print("Escalated:", result.get("escalated", False))
print("Escalation Message:", result.get("escalation_message"))
