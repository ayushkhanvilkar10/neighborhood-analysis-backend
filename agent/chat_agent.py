"""
Chat Agent — ReAct
==================
A streaming ReAct agent with three Boston Open Data tools.

Architecture:
    START → assistant → tools_condition → ToolNode → assistant → ... → END

The agent decides which tools to call based on the user's question — it only
fetches the data that is actually relevant to the query rather than running
all tools on every turn.

Tools:
    - fetch_311(neighborhood)          → top 311 complaint types
    - fetch_crime(neighborhood, street) → top crime types on a street
    - fetch_property(zip_code)          → property type breakdown

No checkpointer — history is owned by Supabase and reconstructed by the
WebSocket handler on every invocation.

Streaming via astream_events (v2) — tokens from the assistant node are
yielded as they arrive and piped over a WebSocket.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ─────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────

BOSTON_API_URL = "https://data.boston.gov/api/3/action/datastore_search_sql"

NEIGHBORHOOD_TO_DISTRICT: dict[str, str] = {
    "Allston":                                        "D14",
    "Allston / Brighton":                             "D14",
    "Back Bay":                                       "D4",
    "Beacon Hill":                                    "A1",
    "Brighton":                                       "D14",
    "Charlestown":                                    "A15",
    "Dorchester":                                     "C11",
    "Downtown / Financial District":                  "A1",
    "East Boston":                                    "A7",
    "Fenway / Kenmore / Audubon Circle / Longwood":   "D4",
    "Greater Mattapan":                               "B3",
    "Hyde Park":                                      "E18",
    "Jamaica Plain":                                  "E13",
    "Mattapan":                                       "B3",
    "Mission Hill":                                   "E13",
    "Roslindale":                                     "E13",
    "Roxbury":                                        "B2",
    "South Boston":                                   "C6",
    "South Boston / South Boston Waterfront":         "C6",
    "South End":                                      "D4",
    "West Roxbury":                                   "E5",
}


# ─────────────────────────────────────────────
# 2. Tools
# ─────────────────────────────────────────────

@tool
async def fetch_311(neighborhood: str) -> str:
    """
    Fetch the top 15 most frequent 311 service request types for a Boston
    neighborhood. Use this when the user asks about complaints, noise,
    trash, code enforcement, quality of life issues, or general neighborhood
    conditions.

    Args:
        neighborhood: The neighborhood name exactly as it appears in the
                      Boston 311 dataset, e.g. 'Back Bay', 'Jamaica Plain'.
    """
    sql = (
        'SELECT "type", COUNT(*) as count '
        'FROM "1a0b420d-99f1-4887-9851-990b2a5a6e17" '
        f"WHERE \"neighborhood\" = '{neighborhood}' "
        'GROUP BY "type" ORDER BY count DESC LIMIT 15'
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(BOSTON_API_URL, params={"sql": sql})
            resp.raise_for_status()
            records = resp.json().get("result", {}).get("records", [])
        if not records:
            return f"No 311 records found for neighborhood: {neighborhood}"
        lines = [f"Top 311 requests for {neighborhood}:"]
        for r in records:
            lines.append(f"  • {r['type']}: {r['count']}")
        return "\n".join(lines)
    except httpx.HTTPError as e:
        return f"Error fetching 311 data: {e}"


@tool
async def fetch_crime(neighborhood: str, street: str) -> str:
    """
    Fetch the top 15 most frequent crime types on a specific Boston street
    in the current year. Use this when the user asks about crime, safety,
    incidents, or how safe a particular street is.

    Args:
        neighborhood: The neighborhood name used to resolve the BPD district,
                      e.g. 'Back Bay', 'South Boston'.
        street:       The street name in uppercase as it appears in the BPD
                      dataset, e.g. 'NEWBURY ST', 'TREMONT ST'.
    """
    district = NEIGHBORHOOD_TO_DISTRICT.get(neighborhood)
    if not district:
        return f"Could not resolve a BPD district for neighborhood: '{neighborhood}'."
    sql = (
        'SELECT "OFFENSE_DESCRIPTION", COUNT(*) as count '
        'FROM "b973d8cb-eeb2-4e7e-99da-c92938efc9c0" '
        f"WHERE \"DISTRICT\" = '{district}' "
        f"AND \"STREET\" = '{street}' "
        "AND \"YEAR\" = '2026' "
        'GROUP BY "OFFENSE_DESCRIPTION" ORDER BY count DESC LIMIT 15'
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(BOSTON_API_URL, params={"sql": sql})
            resp.raise_for_status()
            records = resp.json().get("result", {}).get("records", [])
        if not records:
            return f"No crime records found for {street} in {neighborhood} (District {district}) in 2026."
        lines = [f"Top crimes on {street} ({neighborhood}, 2026):"]
        for r in records:
            lines.append(f"  • {r['OFFENSE_DESCRIPTION']}: {r['count']}")
        return "\n".join(lines)
    except httpx.HTTPError as e:
        return f"Error fetching crime data: {e}"


@tool
async def fetch_property(zip_code: str) -> str:
    """
    Fetch the top 15 most frequent property use types for a Boston zip code.
    Use this when the user asks about property types, housing mix, condos,
    single families, rental stock, or what kinds of buildings exist in an area.

    Args:
        zip_code: A Boston zip code, e.g. '02116', '02130'.
    """
    sql = (
        'SELECT "LU_DESC", COUNT(*) as count '
        'FROM "ee73430d-96c0-423e-ad21-c4cfb54c8961" '
        f"WHERE \"ZIP_CODE\" = '{zip_code}' "
        'GROUP BY "LU_DESC" ORDER BY count DESC LIMIT 15'
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(BOSTON_API_URL, params={"sql": sql})
            resp.raise_for_status()
            records = resp.json().get("result", {}).get("records", [])
        if not records:
            return f"No property records found for zip code: {zip_code}"
        lines = [f"Property mix for zip code {zip_code}:"]
        for r in records:
            lines.append(f"  • {r['LU_DESC']}: {r['count']}")
        return "\n".join(lines)
    except httpx.HTTPError as e:
        return f"Error fetching property data: {e}"


# ─────────────────────────────────────────────
# 3. LLM with tools bound
# ─────────────────────────────────────────────

tools = [fetch_311, fetch_crime, fetch_property]

model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)


# ─────────────────────────────────────────────
# 4. System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a knowledgeable and direct real estate assistant specializing in "
    "Boston neighborhoods. You have access to live Boston Open Data through "
    "three tools: 311 service requests, crime incidents, and property assessments.\n\n"
    "When a user asks about a neighborhood, street, or area — use the appropriate "
    "tool to fetch real data before answering. Do not answer from general knowledge "
    "alone when data is available.\n\n"
    "Guidelines:\n"
    "- Use fetch_311 for questions about complaints, quality of life, or general "
    "neighborhood conditions.\n"
    "- Use fetch_crime for questions about safety or crime on a specific street. "
    "Always use uppercase for street names (e.g. 'NEWBURY ST').\n"
    "- Use fetch_property for questions about housing types, property mix, or "
    "what kinds of buildings exist in a zip code.\n"
    "- You can call multiple tools in one turn if the question requires it.\n"
    "- Be specific with numbers from the data. Do not soften or hedge findings.\n"
    "- When you don't have enough context (e.g. the user hasn't given a street "
    "or zip code), ask for it before calling a tool."
))


# ─────────────────────────────────────────────
# 5. Assistant node
# ─────────────────────────────────────────────

def call_model(state: MessagesState) -> dict:
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": response}


# ─────────────────────────────────────────────
# 6. Build ReAct graph (no checkpointer — stateless by design)
# ─────────────────────────────────────────────

workflow = StateGraph(MessagesState)

workflow.add_node("assistant", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", tools_condition)
workflow.add_edge("tools", "assistant")

graph = workflow.compile()


# ─────────────────────────────────────────────
# 7. Streaming helper
# ─────────────────────────────────────────────

async def stream_chat(messages: list, config: dict):
    """
    Async generator that yields response tokens one at a time.

    Only tokens from the assistant node are yielded — tool call internals
    are filtered out and never sent to the client.

    Args:
        messages: Full conversation history as a list of LangChain message
                  objects (HumanMessage, AIMessage). The caller reconstructs
                  this from Supabase on every turn.
        config:   RunnableConfig dict — no checkpointer is attached.

    Yields:
        str: Individual tokens from the model's response stream.
    """
    async for event in graph.astream_events(
        {"messages": messages},
        config,
        version="v2",
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node") == "assistant"
        ):
            token = event["data"]["chunk"].content
            if token:
                yield token


# ─────────────────────────────────────────────
# 8. Runnable demo
# ─────────────────────────────────────────────

async def _run_turn(history: list, user_input: str) -> str:
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

    user_input_1 = "What are the most common 311 complaints in Back Bay?"
    response_1 = await _run_turn(history, user_input_1)
    history += [HumanMessage(content=user_input_1), AIMessage(content=response_1)]

    user_input_2 = "What about crime on Newbury St?"
    response_2 = await _run_turn(history, user_input_2)
    history += [HumanMessage(content=user_input_2), AIMessage(content=response_2)]

    user_input_3 = "And what's the property mix in zip code 02116?"
    await _run_turn(history, user_input_3)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
