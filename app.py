"""
Streamlit UI for the Customer Support FAQ Agent.

Run with:
    streamlit run app.py
"""

import streamlit as st
from agent import SupportAgent

st.set_page_config(page_title="Support Agent", page_icon="💬")
st.title("💬 Acme SaaS Support")
st.caption("RAG-based FAQ agent with tool-calling (order lookup, escalation)")

if "agent" not in st.session_state:
    try:
        st.session_state.agent = SupportAgent()
    except Exception as e:
        st.error(f"Couldn't initialize agent — check your ANTHROPIC_API_KEY. ({e})")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = st.session_state.agent.send(prompt)
        st.markdown(reply)
    st.session_state.messages.append(("assistant", reply))
