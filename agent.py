import os
import sqlite3
from pathlib import Path

import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools

Path("data").mkdir(exist_ok=True)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

ALLOWED_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
}

SYSTEM_PROMPT = """
You are BappyGPT, a helpful AI assistant.

You should answer questions normally.

Use tools ONLY when they are actually required.

Available tools:
- Calculator
- Web Search
- Memory
- Document Search

For normal conversation, coding, explanations, writing blogs,
summaries, essays, etc., answer directly without using any tool.

Use web search only for latest/current information.
Use calculator only for calculations.
Use memory tools only when asked.
Use document search only when user asks about uploaded files.
"""


def normalize_model_name(model_name: str |None):
    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


def build_agent(model_name):

    selected_model = normalize_model_name(model_name)

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.3,
        streaming=True,
    )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):

        messages = state["messages"]

        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + messages
        )

        return {
            "messages": [response]
        }

    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "chatbot")

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(conn)

    return workflow.compile(
        checkpointer=checkpointer
    )


_AGENT_CACHE = {}


def get_agent(model_name=None):

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]