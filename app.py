from langchain_core.messages import HumanMessage
from agent import get_agent
from database import init_db

init_db()

agent = get_agent("gemini-2.5-flash")

config = {
    "configurable": {
        "thread_id": "test_thread"
    }
}

for message, metadata in agent.stream(
    {
        "messages": [
            HumanMessage(
                content="What is my name?"
            )
        ]
    },
    config=config,
    stream_mode="messages",
):
    if hasattr(message, "content") and message.content:
        print(message.content, end="", flush=True)