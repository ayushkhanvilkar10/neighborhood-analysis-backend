"""
Boston Neighborhood Analysis Agent
===================================
Async parallelized graph: three fetch nodes write to a shared `context` list
(using operator.add reducer), then one summarize node reads from it.

Architecture:
    START → [fetch_311, fetch_crime, fetch_property] (parallel) → summarize → END
"""

import operator
import httpx
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import START, END, StateGraph


# ─────────────────────────────────────────────
# 1. Neighborhood → BPD District mapping
# ─────────────────────────────────────────────
NEIGHBORHOOD_TO_DISTRICT = {
    "Allston":                                          "D14",
    "Allston / Brighton":                               "D14",
    "Back Bay":                                         "D4",
    "Beacon Hill":                                      "A1",
    "Brighton":                                         "D14",
    "Charlestown":                                      "A15",
    "Dorchester":                                       "C11",
    "Downtown / Financial District":                    "A1",
    "East Boston":                                      "A7",
    "Fenway / Kenmore / Audubon Circle / Longwood":     "D4",
    "Greater Mattapan":                                 "B3",
    "Hyde Park":                                        "E18",
    "Jamaica Plain":                                    "E13",
    "Mattapan":                                         "B3",
    "Mission Hill":                                     "E13",
    "Roslindale":                                       "E13",
    "Roxbury":                                          "B2",
    "South Boston":                                     "C6",
    "South Boston / South Boston Waterfront":           "C6",
    "South End":                                        "D4",
    "West Roxbury":                                     "E5",
}


# ─────────────────────────────────────────────
# 2. State schemas
# ─────────────────────────────────────────────

class State(TypedDict):
    # Inputs
    neighborhood: str
    street_name: str
    zip_code: str
    # Parallel fetch nodes all append to this list (operator.add reducer)
    context: Annotated[list[str], operator.add]


class OutputState(TypedDict):
    requests_311:    str
    crime_safety:    str
    property_mix:    str
    overall_verdict: str


class NeighborhoodReport(TypedDict):
    requests_311:    str
    crime_safety:    str
    property_mix:    str
    overall_verdict: str


# ─────────────────────────────────────────────
# 3. LLM
# ─────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o")
llm_with_structure = llm.with_structured_output(NeighborhoodReport)


# ─────────────────────────────────────────────
# 4. Async fetch nodes (run in parallel)
# ─────────────────────────────────────────────

BOSTON_API_URL = "https://data.boston.gov/api/3/action/datastore_search_sql"


async def fetch_311(state: State) -> dict:
    """Fetch top 15 311 request types for the neighborhood."""
    neighborhood = state["neighborhood"]
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
            return {"context": [f"No 311 records found for neighborhood: {neighborhood}"]}
        lines = [f"Top 311 requests for {neighborhood}:\n"]
        for r in records:
            lines.append(f"  • {r['type']}: {r['count']}")
        return {"context": ["\n".join(lines)]}
    except httpx.HTTPError as e:
        return {"context": [f"Error fetching 311 data: {e}"]}


async def fetch_crime(state: State) -> dict:
    """Fetch top 15 crime types for the street, resolving district from neighborhood."""
    neighborhood = state["neighborhood"]
    street_name  = state["street_name"]
    district = NEIGHBORHOOD_TO_DISTRICT.get(neighborhood)
    if not district:
        return {"context": [f"Could not resolve a BPD district for neighborhood: '{neighborhood}'."]}
    sql = (
        'SELECT "OFFENSE_DESCRIPTION", COUNT(*) as count '
        'FROM "b973d8cb-eeb2-4e7e-99da-c92938efc9c0" '
        f"WHERE \"DISTRICT\" = '{district}' "
        f"AND \"STREET\" = '{street_name}' "
        "AND \"YEAR\" = '2026' "
        'GROUP BY "OFFENSE_DESCRIPTION" ORDER BY count DESC LIMIT 15'
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(BOSTON_API_URL, params={"sql": sql})
            resp.raise_for_status()
            records = resp.json().get("result", {}).get("records", [])
        if not records:
            return {"context": [f"No crime records found for street: {street_name} in district {district} ({neighborhood})"]}
        lines = [f"Top crimes on {street_name} (District {district} – {neighborhood}, 2026):\n"]
        for r in records:
            lines.append(f"  • {r['OFFENSE_DESCRIPTION']}: {r['count']}")
        return {"context": ["\n".join(lines)]}
    except httpx.HTTPError as e:
        return {"context": [f"Error fetching crime data: {e}"]}


async def fetch_property(state: State) -> dict:
    """Fetch top 15 property types for the zip code."""
    zip_code = state["zip_code"]
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
            return {"context": [f"No property records found for zip code: {zip_code}"]}
        lines = [f"Property mix for zip code {zip_code}:\n"]
        for r in records:
            lines.append(f"  • {r['LU_DESC']}: {r['count']}")
        return {"context": ["\n".join(lines)]}
    except httpx.HTTPError as e:
        return {"context": [f"Error fetching property data: {e}"]}


# ─────────────────────────────────────────────
# 5. Async summarization node
# ─────────────────────────────────────────────

sys_msg = SystemMessage(content=(
    "You are a straight-talking neighborhood-analysis assistant for Boston house hunters. "
    "Your job is to give honest, specific, buyer-focused insights — not reassuring generalities. "
    "A buyer making a major financial decision needs the full picture, including red flags.\n\n"
    "You will receive three raw data blocks (311 complaints, crime incidents, property mix). "
    "Produce a structured report with exactly four fields as defined.\n\n"
    "## 311 Service Requests\n"
    "  - 'CE Collection' = Code Enforcement Collection (city pursuing code violations — flag this).\n"
    "  - 'Needle Pickup' and 'Encampments' = visible substance-abuse or homelessness — state plainly.\n"
    "  - 'Unsatisfactory Living Conditions' and 'Heat - Excessive/Insufficient' = housing quality issues.\n"
    "  - 'Space Savers' = residents blocking public parking after snowstorms — sign of parking scarcity.\n"
    "  - High 'Illegal Dumping' or 'Abandoned Vehicles' = neighborhood neglect.\n\n"
    "## Crime & Safety\n"
    "  - Context matters: commercial streets have higher expected counts than residential ones.\n"
    "  - Distinguish procedural incidents (Investigate Person/Property) from actual crimes.\n"
    "  - Flag auto theft, drug offenses, robbery, threats explicitly — do not minimize.\n"
    "  - Low counts on a quiet residential street are reassuring — say so.\n\n"
    "## Property Mix\n"
    "  - 'CONDO PARKING (RES)' = parking sold separately — extra cost for buyers.\n"
    "  - 'SUBSD HOUSING S-8' = Section 8 subsidized housing.\n"
    "  - 'CITY OF BOSTON' / 'BOST REDEVELOP AUTH' / 'BOS HOUSING AUTHOR' = publicly owned land.\n"
    "  - 'RES LAND (Unusable)' = vacant lots — possible blight or future development.\n"
    "  - Two- and three-family home dominance = rental-heavy neighborhood.\n\n"
    "## Overall Verdict\n"
    "Synthesize across all three datasets. Connect the dots. Be specific and direct."
))


async def summarize(state: State) -> dict:
    """Call the LLM with all fetched context and return the four structured fields."""
    user_msg = HumanMessage(content=(
        f"Neighborhood: {state['neighborhood']}\n"
        f"Street: {state['street_name']}\n"
        f"Zip Code: {state['zip_code']}\n\n"
        + "\n\n".join(state["context"])
    ))
    report = await llm_with_structure.ainvoke([sys_msg, user_msg])
    return {
        "requests_311":    report["requests_311"],
        "crime_safety":    report["crime_safety"],
        "property_mix":    report["property_mix"],
        "overall_verdict": report["overall_verdict"],
    }


# ─────────────────────────────────────────────
# 6. Build the parallelized graph
# ─────────────────────────────────────────────

builder = StateGraph(State, output_schema=OutputState)

builder.add_node("fetch_311",      fetch_311)
builder.add_node("fetch_crime",    fetch_crime)
builder.add_node("fetch_property", fetch_property)
builder.add_node("summarize",      summarize)

builder.add_edge(START,          "fetch_311")
builder.add_edge(START,          "fetch_crime")
builder.add_edge(START,          "fetch_property")

builder.add_edge("fetch_311",      "summarize")
builder.add_edge("fetch_crime",    "summarize")
builder.add_edge("fetch_property", "summarize")

builder.add_edge("summarize", END)

graph = builder.compile()
