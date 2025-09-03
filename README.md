**Support Ticket Resolution Agent**


This project is a Support Ticket Resolution Agent built with LangGraph and Gemini.
It simulates how real companies handle support tickets: classify → draft → review → retry → escalate.

*Workflow*

Classify ticket into Billing, Technical, Security, or General.

Retrieve knowledge base context.

Draft a reply with Gemini.

Review the draft (check rules: polite, safe, no refund promises).

Retry once if rejected.

Escalate to human after 2 failed attempts.

*Setup*
```bash
git clone https://github.com/YOURNAME/support-ticket-agent.git
cd support-ticket-agent
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux
pip install -r requirements.txt
```

Set API key:

export GEMINI_API_KEY="your-key"   # Mac/Linux
$env:GEMINI_API_KEY="your-key"     # Windows


*Run dev server*
```bash
python -m langgraph dev graph:app

Test

Example request (Windows PowerShell):

Invoke-RestMethod -Uri http://localhost:8123/invoke `
  -Method Post `
  -Body '{"subject": "Refund request", "description": "I was charged twice"}' `
  -ContentType "application/json"


Expected: Ticket → classified as Billing → draft generated → reviewed → retried or escalated.
