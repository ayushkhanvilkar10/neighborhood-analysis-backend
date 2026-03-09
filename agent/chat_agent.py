"""
Chat Agent
==========
A streaming conversational agent built on LangGraph's MessagesState.

Architecture:
    START → conversation → END

Design decisions:
- No checkpointer: conversation history is owned by Supabase (chat_messages
  table) and reconstructed by the caller before each invocation. This keeps
  the graph stateless and the persistence layer fully under our control.
- No summarization: history management (truncation, summarization) will be
  added later once the end-to-end WebSocket flow is in place.
- Streaming via astream_events (v2) + on_chat_model_stream events: tokens
  are yielded as they arrive, suitable for piping directly over a WebSocket.

Tools:
- Tool support will be added in a follow-up iteration. The graph is structured
  so that a ToolNode and tools_condition edge can be dropped in with minimal
  changes.

Usage (WebSocket handler, simplified):
    from agent.chat_agent import stream_chat

    async for token in stream_chat(messages, config):
        await websocket.send_text(token)
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END


# ─────────────────────────────────────────────
# 1. State
# ─────────────────────────────────────────────

# MessagesState provides a single `messages` key with an append-only reducer.
# The caller is responsible for passing the full history on every invocation.
State = MessagesState


# ─────────────────────────────────────────────
# 2. LLM
# ─────────────────────────────────────────────

model = ChatOpenAI(model="gpt-4o", temperature=0)


# ─────────────────────────────────────────────
# 3. System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a knowledgeable and direct real estate assistant specializing in "
    "Boston neighborhoods. You help buyers understand neighborhood data — crime, "
    "property mix, transit, green space, and more. You give honest, specific "
    "answers grounded in data. When you don't have data, say so plainly."
))


# ─────────────────────────────────────────────
# 4. Conversation node
# ─────────────────────────────────────────────

def call_model(state: State, config: RunnableConfig) -> dict:
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = model.invoke(messages, config)
    return {"messages": response}


# ─────────────────────────────────────────────
# 5. Build graph (no checkpointer — stateless by design)
# ─────────────────────────────────────────────

workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_edge(START, "conversation")
workflow.add_edge("conversation", END)

graph = workflow.compile()


# ─────────────────────────────────────────────
# 6. Streaming helper
# ─────────────────────────────────────────────

async def stream_chat(messages: list, config: dict):
    """
    Async generator that yields response tokens one at a time.

    Args:
        messages: Full conversation history as a list of LangChain message
                  objects (HumanMessage, AIMessage). The caller reconstructs
                  this from Supabase on every turn.
        config:   RunnableConfig — can carry thread_id or other metadata
                  but no checkpointer is attached to this graph.

    Yields:
        str: Individual tokens from the model's response stream.

    Example:
        from langchain_core.messages import HumanMessage, AIMessage

        history = [
            HumanMessage(content="Tell me about Back Bay."),
            AIMessage(content="Back Bay is a dense, walkable neighborhood..."),
        ]
        new_message = HumanMessage(content="What about crime there?")

        async for token in stream_chat(history + [new_message], config={}):
            await websocket.send_text(token)
    """
    async for event in graph.astream_events(
        {"messages": messages},
        config,
        version="v2",
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node") == "conversation"
        ):
            token = event["data"]["chunk"].content
            if token:
                yield token


# ─────────────────────────────────────────────
# 7. Runnable demo (two-turn conversation)
# ─────────────────────────────────────────────

async def _run_turn(history: list, user_input: str) -> str:
    """Stream a single turn, print tokens as they arrive, return full response."""
    from langchain_core.messages import HumanMessage, AIMessage

    new_message = HumanMessage(content=user_input)
    all_messages = history + [new_message]

    print(f"\nYou: {user_input}")
    print("AI: ", end="", flush=True)

    full_response = ""
    async for token in stream_chat(all_messages, config={}):
        print(token, end="", flush=True)
        full_response += token

    print()
    return full_response


async def _main():
    from langchain_core.messages import HumanMessage, AIMessage

    history = []

    user_input_1 = "What makes Back Bay a desirable neighborhood in Boston?"
    response_1 = await _run_turn(history, user_input_1)
    history += [
        HumanMessage(content=user_input_1),
        AIMessage(content=response_1),
    ]

    user_input_2 = "How does that compare to South Boston?"
    await _run_turn(history, user_input_2)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
