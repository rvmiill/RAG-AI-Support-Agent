# Customer Support FAQ Agent

A RAG-based customer support agent built on the Claude API. Retrieves relevant
FAQ entries via TF-IDF search, then lets Claude answer, call tools (order
lookup, escalation), or ask clarifying questions.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

**Command line:**
```bash
python agent.py
```

**Web UI (Streamlit):**
```bash
streamlit run app.py
```

## How it works

1. **`faq_data.json`** — your knowledge base. Each entry has a question and
   answer. Replace the sample entries with your own product's FAQs.
2. **`agent.py`**
   - `FAQRetriever` — TF-IDF + cosine similarity search over the FAQ file.
     Good for up to a few hundred entries. For a larger knowledge base,
     swap this out for a real vector store (Chroma, Pinecone, pgvector) with
     embeddings.
   - `SupportAgent` — sends the user's message plus retrieved FAQ context to
     Claude, with two tools available: `look_up_order_status` and
     `escalate_to_human`. Currently both are mocked — wire them to your real
     order system / ticketing system (Zendesk, Freshdesk, etc.) in production.
3. **`app.py`** — a minimal Streamlit chat interface on top of the agent.

## Extending this

- **Bigger knowledge base / real embeddings:** replace `FAQRetriever` with a
  vector DB and an embeddings model (e.g. `voyage-3` via Anthropic's
  recommended embeddings partner, or OpenAI embeddings).
- **More tools:** add entries to `TOOLS` and `TOOL_IMPL` in `agent.py` —
  e.g. `update_shipping_address`, `apply_refund`, `check_subscription_status`.
- **Conversation memory across sessions:** persist `agent.history` per user
  (e.g. keyed by session/user ID) in a database instead of Streamlit's
  in-memory session state.
- **Guardrails:** add a check before escalation/refund tools fire (e.g.
  confirm with the user first) since those have real-world side effects.
