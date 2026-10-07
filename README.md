Student Records Agent - LangChain + LangGraph + Groq (free tier) + SQLite

Setup:
    pip install -U langchain-groq langchain-core langgraph
    export GROQ_API_KEY="gsk_..."        # free key from https://console.groq.com/keys
    python student_agent.py

Everything is free/local: Groq free-tier LLM + a local SQLite file. No paid API.
The LLM never writes raw SQL; it only calls the safe, parameterized tools below.
