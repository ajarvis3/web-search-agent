"""
Web-search agent backed by Google Gemini and LangChain.

The agent is given a DuckDuckGo search tool.  The Gemini LLM decides
for each query whether its built-in knowledge is sufficient or whether
it needs to call the search tool to find up-to-date information.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions accurately. "
    "You have access to a web_search tool. "
    "First, consider whether you already know the answer from your training data. "
    "If the answer is well-established and not time-sensitive, answer directly "
    "without using the tool. "
    "If the answer may be outdated, requires current information, or you are "
    "uncertain, use the web_search tool to look it up before answering."
)


def _build_agent() -> Any:
    """Construct and return a compiled LangChain agent graph."""
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    tools = [DuckDuckGoSearchRun(name="web_search")]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
    )


_agent: Any = None


def get_agent() -> Any:
    """Return the singleton agent graph, creating it on first call."""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def run_agent(query: str) -> str:
    """Run the agent for *query* and return its final answer string."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    agent = get_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query.strip()}]}
    )
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return ""
