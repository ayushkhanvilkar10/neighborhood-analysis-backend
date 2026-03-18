"""
Chat Agent — ReAct
==================
A streaming ReAct agent with eight Boston Open Data tools.

Architecture:
    START → assistant → tools_condition → ToolNode → assistant → ... → END

The agent decides which tools to call based on the user's question — it only
fetches the data that is actually relevant to the query rather than running
all tools on every turn.

Tools:
    - fetch_311(neighborhood)           → top 311 complaint types
    - fetch_crime(neighborhood, street)  → top crime types on a street
    - fetch_property(zip_code)           → property type breakdown
    - fetch_permits(zip_code)            → building permit activity
    - fetch_entertainment(zip_code)      → entertainment license breakdown
    - fetch_traffic_safety(street)       → crash and fatality data
    - fetch_gun_violence(neighborhood)   → shooting victims and shots fired

No checkpointer — history is owned by Supabase and reconstructed by the
WebSocket handler on every invocation.

Streaming via astream_events (v2) — tokens from the assistant node are
yielded as they arrive and piped over a WebSocket.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

import asyncio
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


@tool
async def fetch_permits(zip_code: str) -> str:
    """
    Fetch building permit activity for a Boston zip code over the past two
    years, ordered by total declared investment value. Use this when the user
    asks about development, construction, renovation, new buildings, investment
    trends, or how much a neighborhood is changing.

    Args:
        zip_code: A Boston zip code, e.g. '02116', '02130'.
    """
    sql = (
        'SELECT "worktype", COUNT(*) as count, '
        'SUM(CAST(REPLACE(REPLACE("declared_valuation", \'$\', \'\'), \',\', \'\') AS NUMERIC)) as total_value '
        'FROM "6ddcd912-32a0-43df-9908-63574f8c7e77" '
        f"WHERE \"zip\" = '{zip_code}' "
        "AND \"issued_date\" >= (CURRENT_DATE - INTERVAL '2 years') "
        "AND \"declared_valuation\" != '' "
        'GROUP BY "worktype" '
        'ORDER BY total_value DESC '
        'LIMIT 15'
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(BOSTON_API_URL, params={"sql": sql})
            resp.raise_for_status()
            records = resp.json().get("result", {}).get("records", [])
        if not records:
            return f"No building permit records found for zip code: {zip_code}"
        lines = [f"Building permit activity for zip code {zip_code} (past 2 years, ordered by total investment value):"]
        for r in records:
            worktype = r["worktype"] or "UNKNOWN"
            total_value = float(r["total_value"] or 0)
            lines.append(f"  • {worktype}: {r['count']} permits — ${total_value:,.0f} total declared value")
        return "\n".join(lines)
    except httpx.HTTPError as e:
        return f"Error fetching permit data: {e}"


@tool
async def fetch_entertainment(zip_code: str) -> str:
    """
    Fetch active entertainment license types for a Boston zip code. Use this
    when the user asks about nightlife, bars, restaurants, noise levels, live
    music, DJs, or the entertainment and dining scene in a neighborhood.

    Args:
        zip_code: A Boston zip code, e.g. '02116', '02130'.
    """
    sql = (
        'SELECT "unit_type", COUNT(*) as count, '
        'SUM(CAST("numberofunits" AS INTEGER)) as total_units '
        'FROM "1c4c1f7c-9a2a-4f4f-85a7-d3462c6bc9cb" '
        f"WHERE \"zip\" = '{zip_code}' "
        "AND \"status\" = 'Active' "
        "AND \"numberofunits\" != '' "
        'GROUP BY "unit_type" '
        'ORDER BY count DESC '
        'LIMIT 15'
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(BOSTON_API_URL, params={"sql": sql})
            resp.raise_for_status()
            records = resp.json().get("result", {}).get("records", [])
        if not records:
            return f"No entertainment license records found for zip code: {zip_code}"
        lines = [f"Entertainment license breakdown for zip code {zip_code}:"]
        for r in records:
            lines.append(f"  • {r['unit_type']}: {r['count']} venues, {r['total_units']} total units")
        return "\n".join(lines)
    except httpx.HTTPError as e:
        return f"Error fetching entertainment data: {e}"


@tool
async def fetch_traffic_safety(street: str) -> str:
    """
    Fetch traffic crash and fatality data for a specific Boston street since
    2022. Use this when the user asks about traffic safety, crashes, pedestrian
    safety, cyclist safety, dangerous intersections, or fatalities.

    Args:
        street: The street name in uppercase as it appears in the Vision Zero
                dataset, e.g. 'NEWBURY ST', 'COMMONWEALTH AVE'.
    """
    street_filter = (
        f"(\"street\" = '{street}' OR \"xstreet1\" = '{street}' OR \"xstreet2\" = '{street}')"
    )
    sql_crash_modes = (
        'SELECT "mode_type", COUNT(*) as count '
        'FROM "e4bfe397-6bfc-49c5-9367-c879fac7401d" '
        f'WHERE {street_filter} '
        "AND \"dispatch_ts\" >= '2022-01-01' "
        'GROUP BY "mode_type" ORDER BY count DESC'
    )
    sql_hotspots = (
        'SELECT "xstreet1", "xstreet2", COUNT(*) as count '
        'FROM "e4bfe397-6bfc-49c5-9367-c879fac7401d" '
        f'WHERE {street_filter} '
        "AND \"dispatch_ts\" >= '2022-01-01' "
        'GROUP BY "xstreet1", "xstreet2" ORDER BY count DESC LIMIT 5'
    )
    sql_fatalities = (
        'SELECT "mode_type", "street", "xstreet1", "xstreet2", "date_time" '
        'FROM "92f18923-d4ec-4c17-9405-4e0da63e1d6c" '
        f'WHERE {street_filter} ORDER BY "date_time" DESC'
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp_modes, resp_hotspots, resp_fatalities = await asyncio.gather(
                client.get(BOSTON_API_URL, params={"sql": sql_crash_modes}),
                client.get(BOSTON_API_URL, params={"sql": sql_hotspots}),
                client.get(BOSTON_API_URL, params={"sql": sql_fatalities}),
            )
        crash_modes = resp_modes.json().get("result", {}).get("records", [])
        hotspots    = resp_hotspots.json().get("result", {}).get("records", [])
        fatalities  = resp_fatalities.json().get("result", {}).get("records", [])
        if not crash_modes and not fatalities:
            return f"No traffic safety records found for street: {street}"
        lines = [f"Traffic safety data for {street} (since 2022):"]
        mode_labels = {"mv": "Motor vehicle", "ped": "Pedestrian", "bike": "Cyclist"}
        if crash_modes:
            total = sum(int(r["count"]) for r in crash_modes)
            lines.append(f"\nCrashes (total: {total}):")
            for r in crash_modes:
                lines.append(f"  • {mode_labels.get(r['mode_type'], r['mode_type'])}: {r['count']}")
        if hotspots:
            lines.append("\nTop crash intersections:")
            for r in hotspots:
                x1, x2 = r["xstreet1"] or "", r["xstreet2"] or ""
                intersection = f"{x1} & {x2}" if x1 and x2 else x1 or x2
                lines.append(f"  • {intersection}: {r['count']} crashes")
        if fatalities:
            lines.append(f"\nFatalities (all time — {len(fatalities)} total):")
            for r in fatalities:
                label = mode_labels.get(r["mode_type"], r["mode_type"])
                x1, x2 = r["xstreet1"] or "", r["xstreet2"] or ""
                intersection = f"{x1} & {x2}" if x1 and x2 else (r["street"] or "unknown")
                date = r["date_time"][:10] if r["date_time"] else "unknown date"
                lines.append(f"  • {label} fatality — {intersection} — {date}")
        else:
            lines.append("\nNo fatalities on record for this street.")
        return "\n".join(lines)
    except httpx.HTTPError as e:
        return f"Error fetching traffic safety data: {e}"


@tool
async def fetch_gun_violence(neighborhood: str) -> str:
    """
    Fetch shooting victim and shots fired data for a Boston neighborhood's
    BPD district since 2022. Use this when the user asks about gun violence,
    shootings, shots fired, or district-level gun safety concerns.

    Args:
        neighborhood: The neighborhood name used to resolve the BPD district,
                      e.g. 'Back Bay', 'Roxbury', 'South Boston'.
    """
    district = NEIGHBORHOOD_TO_DISTRICT.get(neighborhood)
    if not district:
        return f"Could not resolve a BPD district for neighborhood: '{neighborhood}'."
    sql_shootings = (
        'SELECT "shooting_type_v2", COUNT(*) as count '
        'FROM "73c7e069-701f-4910-986d-b950f46c91a1" '
        f"WHERE \"district\" = '{district}' "
        "AND \"shooting_date\" >= '2022-01-01' "
        'GROUP BY "shooting_type_v2" ORDER BY count DESC'
    )
    sql_shots_fired = (
        'SELECT COUNT(*) as total_shots_fired, '
        'SUM(CASE WHEN "ballistics_evidence" = \'t\' THEN 1 ELSE 0 END) as confirmed_with_ballistics '
        'FROM "c1e4e6ac-8a84-4b48-8a23-7b2645a32ede" '
        f"WHERE \"district\" = '{district}' "
        "AND \"incident_date\" >= '2022-01-01'"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp_shootings, resp_shots = await asyncio.gather(
                client.get(BOSTON_API_URL, params={"sql": sql_shootings}),
                client.get(BOSTON_API_URL, params={"sql": sql_shots_fired}),
            )
        shootings  = resp_shootings.json().get("result", {}).get("records", [])
        shots_data = resp_shots.json().get("result", {}).get("records", [])
        lines = [f"Gun violence data for {neighborhood} (District {district}) since 2022:"]
        if shootings:
            total = sum(int(r["count"]) for r in shootings)
            lines.append(f"\nShooting victims (total: {total}):")
            for r in shootings:
                lines.append(f"  • {r['shooting_type_v2']}: {r['count']}")
        else:
            lines.append("\nNo shooting victim records found since 2022.")
        if shots_data:
            r = shots_data[0]
            lines.append(
                f"\nShots fired incidents: {r['total_shots_fired']} total, "
                f"{r['confirmed_with_ballistics']} confirmed with ballistics"
            )
        return "\n".join(lines)
    except httpx.HTTPError as e:
        return f"Error fetching gun violence data: {e}"


# ─────────────────────────────────────────────
# 3. LLM with tools bound
# ─────────────────────────────────────────────

tools = [
    fetch_311,
    fetch_crime,
    fetch_property,
    fetch_permits,
    fetch_entertainment,
    fetch_traffic_safety,
    fetch_gun_violence,
]

model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)


# ─────────────────────────────────────────────
# 4. System prompt
# ─────────────────────────────────────────────

# Canonical neighborhood → zip code mapping
# Used by the LLM to resolve zip codes without asking the user
NEIGHBORHOOD_ZIP_CODES = """
Allston: 02134
Allston / Brighton: 02134, 02135
Back Bay: 02116
Beacon Hill: 02108, 02114
Brighton: 02135
Charlestown: 02129
Dorchester: 02121, 02122, 02124, 02125
Downtown / Financial District: 02109, 02110, 02113
East Boston: 02128
Fenway / Kenmore / Audubon Circle / Longwood: 02115, 02215
Greater Mattapan: 02126
Hyde Park: 02136
Jamaica Plain: 02130
Mattapan: 02126
Mission Hill: 02120
Roslindale: 02131
Roxbury: 02119
South Boston: 02127
South Boston / South Boston Waterfront: 02127, 02210
South End: 02118
West Roxbury: 02132
"""

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a knowledgeable and direct real estate assistant specializing in "
    "Boston neighborhoods. You have access to live Boston Open Data through "
    "seven tools: 311 service requests, crime incidents, property assessments, "
    "building permits, entertainment licenses, traffic safety, and gun violence.\n\n"
    "When a user asks about a neighborhood, street, or area — use the appropriate "
    "tool to fetch real data before answering. Do not answer from general knowledge "
    "alone when data is available.\n\n"
    "## Zip Code Resolution\n"
    "Many tools require a zip code. Use the following neighborhood → zip code mapping "
    "to resolve zip codes yourself without asking the user:\n"
    f"{NEIGHBORHOOD_ZIP_CODES}\n"
    "Rules:\n"
    "- If a neighborhood maps to exactly one zip code, use it directly and call the tool immediately.\n"
    "- If a neighborhood maps to multiple zip codes, briefly tell the user which areas each "
    "zip code covers and ask them to pick one. Example: Back Bay has one zip (02116) — use it "
    "directly. Dorchester has four zips — ask which part.\n"
    "- If a user mentions a neighborhood by name, resolve its zip code from the mapping above "
    "before deciding whether to ask. Only ask if there is genuine ambiguity.\n\n"
    "## Tool Guidelines\n"
    "- Use fetch_311 for questions about complaints, quality of life, code enforcement, "
    "noise, trash, or general neighborhood conditions.\n"
    "- Use fetch_crime for questions about safety or crime on a specific street. "
    "Always use uppercase for street names (e.g. 'NEWBURY ST').\n"
    "- Use fetch_property for questions about housing types, property mix, condos, "
    "single families, or rental stock in a zip code.\n"
    "- Use fetch_permits for questions about development activity, construction, "
    "renovation, new buildings, or investment trends.\n"
    "- Use fetch_entertainment for questions about nightlife, bars, restaurants, "
    "noise levels, or the entertainment scene.\n"
    "- Use fetch_traffic_safety for questions about traffic crashes, pedestrian "
    "safety, cyclist safety, dangerous intersections, or fatalities on a street. "
    "Always use uppercase for street names.\n"
    "- Use fetch_gun_violence for questions about shootings, gun violence, or "
    "district-level safety concerns.\n"
    "- You can call multiple tools in one turn if the question requires it.\n"
    "- Be specific with numbers from the data. Do not soften or hedge findings.\n"
    "- Only ask the user for information you genuinely cannot resolve yourself. "
    "Zip codes for single-zip neighborhoods should never require asking."
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
    asyncio.run(_main())
