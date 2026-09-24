"""
Customer Support FAQ Agent
--------------------------
A RAG-based support agent built on the Anthropic Claude API.

How it works:
1. Retrieval: TF-IDF similarity search over a local FAQ knowledge base
   (no external embedding API needed — good enough for a few hundred FAQs;
   swap in a real vector DB like Chroma/Pinecone if your KB grows large).
2. Reasoning: Claude decides whether to answer from retrieved FAQs, call a
   tool (e.g. look up an order, escalate to a human), or ask a clarifying
   question.
3. Tools: mocked here (order lookup, escalation ticket) — wire these to
   your real backend/CRM in production.

Set your API key as an environment variable before running:
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import json
import os
from pathlib import Path

import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "claude-sonnet-4-6"
DATA_PATH = Path(__file__).parent / "faq_data.json"
TOP_K = 3


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class FAQRetriever:
    """Simple TF-IDF retriever over the FAQ knowledge base."""

    def __init__(self, data_path: Path = DATA_PATH):
        with open(data_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        # Index over "question + answer" so we match on both phrasing and content
        corpus = [f"{item['question']} {item['answer']}" for item in self.faqs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = TOP_K):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked_idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in ranked_idx:
            if scores[i] > 0:  # skip zero-relevance matches
                results.append({**self.faqs[i], "score": float(scores[i])})
        return results


# ---------------------------------------------------------------------------
# Mock backend tools (replace with real API/CRM/order-system calls)
# ---------------------------------------------------------------------------

def look_up_order_status(order_id: str) -> dict:
    mock_orders = {
        "ORD-1001": {"status": "Shipped", "eta": "2026-09-27"},
        "ORD-1002": {"status": "Processing", "eta": "2026-09-30"},
    }
    return mock_orders.get(order_id, {"status": "Not found", "eta": None})


def escalate_to_human(reason: str, urgency: str = "normal") -> dict:
    # In production: create a ticket in Zendesk/Freshdesk/etc.
    ticket_id = "TICKET-" + str(abs(hash(reason)) % 10000)
    return {"ticket_id": ticket_id, "reason": reason, "urgency": urgency, "status": "created"}


TOOLS = [
    {
        "name": "look_up_order_status",
        "description": "Look up the shipping/processing status of a customer's order by order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID, e.g. ORD-1001"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Create a support ticket and escalate the conversation to a human agent, for issues the FAQ knowledge base can't resolve.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Short summary of why this needs human help"},
                "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
            },
            "required": ["reason"],
        },
    },
]

TOOL_IMPL = {
    "look_up_order_status": lambda **kw: look_up_order_status(**kw),
    "escalate_to_human": lambda **kw: escalate_to_human(**kw),
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly, concise customer support agent for "Acme SaaS".

You will be given relevant FAQ entries retrieved from the knowledge base for
the user's question — use them as your source of truth and answer directly
from them when they apply. Do not make up policies or pricing that aren't in
the FAQs.

If the retrieved FAQs don't cover the question, or the user needs something
account-specific (like an order status), use the available tools.

If you can't resolve the issue with the FAQs or tools, escalate to a human
using the escalate_to_human tool rather than guessing.

Keep answers short and direct — this is a chat interface, not an essay."""


class SupportAgent:
    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.retriever = FAQRetriever()
        self.history: list[dict] = []

    def _build_context_message(self, user_message: str) -> str:
        hits = self.retriever.search(user_message)
        if not hits:
            return f"[No matching FAQ entries found]\n\nUser question: {user_message}"
        context = "\n\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in hits)
        return f"[Retrieved FAQ context]\n{context}\n\nUser question: {user_message}"

    def send(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": self._build_context_message(user_message)})

        while True:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.history,
            )

            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Final text answer
                return "".join(block.text for block in response.content if block.type == "text")

            # Handle tool calls, feed results back, loop again
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = TOOL_IMPL[block.name]
                    result = fn(**block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )
            self.history.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    agent = SupportAgent()
    print("Support Agent ready. Type 'quit' to exit.\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        reply = agent.send(msg)
        print(f"Agent: {reply}\n")
