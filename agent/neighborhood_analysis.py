"""
Boston Neighborhood Analysis Agent
===================================
Async parallelized graph: six fetch nodes write to a shared `context` list
(using operator.add reducer), then one summarize node reads from it.

Architecture:
    START → [fetch_311, fetch_crime, fetch_property, fetch_permits, fetch_entertainment, fetch_traffic_safety, fetch_gun_violence, fetch_green_space] (parallel) → summarize → END
"""

import asyncio
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

# Neighborhood → BPRD Trees neighborhood name mapping
NEIGHBORHOOD_TO_BPRD = {
    "Allston":                                          "Allston-Brighton",
    "Allston / Brighton":                               "Allston-Brighton",
    "Back Bay":                                         "Back Bay/Beacon Hill",
    "Beacon Hill":                                      "Back Bay/Beacon Hill",
    "Brighton":                                         "Allston-Brighton",
    "Charlestown":                                      "Charlestown",
    "Dorchester":                                       "Dorchester",
    "Downtown / Financial District":                    "Central Boston",
    "East Boston":                                      "East Boston",
    "Fenway / Kenmore / Audubon Circle / Longwood":     "Fenway/Longwood",
    "Greater Mattapan":                                 "Mattapan",
    "Hyde Park":                                        "Hyde Park",
    "Jamaica Plain":                                    "Jamaica Plain",
    "Mattapan":                                         "Mattapan",
    "Mission Hill":                                     "Mission Hill",
    "Roslindale":                                       "Roslindale",
    "Roxbury":                                          "Roxbury",
    "South Boston":                                     "South Boston",
    "South Boston / South Boston Waterfront":           "South Boston",
    "South End":                                        "South End",
    "West Roxbury":                                     "West Roxbury",
}

# Zip code → name + entertainment density tiers (high / moderate / low)
# Tiers are relative to all Boston zip codes for each license category.
# Used in summarize to give the LLM context for raw entertainment license numbers.
ZIP_CODE_INFO = {
    "02108": {"name": "Downtown / Beacon Hill",                     "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "high"},
    "02109": {"name": "Downtown / Financial District (Waterfront)",  "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "high"},
    "02110": {"name": "Downtown / Financial District",              "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "moderate"},
    "02111": {"name": "Downtown / Chinatown",                       "widescreen_tv": "high",     "audio_device": "moderate", "disc_jockey": "moderate"},
    "02113": {"name": "North End",                                  "widescreen_tv": "high",     "audio_device": "moderate", "disc_jockey": "moderate"},
    "02114": {"name": "Beacon Hill / West End",                     "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "high"},
    "02115": {"name": "Fenway / Longwood",                          "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "moderate"},
    "02116": {"name": "Back Bay / Bay Village",                     "widescreen_tv": "high",     "audio_device": "high",     "disc_jockey": "high"},
    "02118": {"name": "South End",                                  "widescreen_tv": "moderate", "audio_device": "high",     "disc_jockey": "low"},
    "02119": {"name": "Roxbury",                                    "widescreen_tv": "low",      "audio_device": "low",      "disc_jockey": "low"},
    "02120": {"name": "Mission Hill",                               "widescreen_tv": "low",      "audio_device": "low",      "disc_jockey": "low"},
    "02121": {"name": "Dorchester (North)",                         "widescreen_tv": "low",      "audio_device": "low",      "disc_jockey": "low"},
    "02122": {"name": "Dorchester (Edward Everett Square)",         "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "moderate"},
    "02124": {"name": "Dorchester (Codman Square / Four Corners)",  "widescreen_tv": "low",      "audio_device": "low",      "disc_jockey": "moderate"},
    "02125": {"name": "Dorchester (Savin Hill / Jones Hill)",       "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "moderate"},
    "02126": {"name": "Mattapan",                                   "widescreen_tv": "low",      "audio_device": "low",      "disc_jockey": "low"},
    "02127": {"name": "South Boston",                               "widescreen_tv": "high",     "audio_device": "high",     "disc_jockey": "high"},
    "02128": {"name": "East Boston",                                "widescreen_tv": "high",     "audio_device": "high",     "disc_jockey": "moderate"},
    "02129": {"name": "Charlestown",                                "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "low"},
    "02130": {"name": "Jamaica Plain",                              "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "moderate"},
    "02131": {"name": "Roslindale",                                 "widescreen_tv": "low",      "audio_device": "low",      "disc_jockey": "low"},
    "02132": {"name": "West Roxbury",                               "widescreen_tv": "low",      "audio_device": "low",      "disc_jockey": "moderate"},
    "02134": {"name": "Allston",                                    "widescreen_tv": "moderate", "audio_device": "high",     "disc_jockey": "moderate"},
    "02135": {"name": "Brighton",                                   "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "moderate"},
    "02136": {"name": "Hyde Park",                                  "widescreen_tv": "moderate", "audio_device": "moderate", "disc_jockey": "moderate"},
    "02210": {"name": "South Boston Waterfront / Seaport",          "widescreen_tv": "high",     "audio_device": "high",     "disc_jockey": "high"},
    "02215": {"name": "Fenway / Kenmore",                           "widescreen_tv": "high",     "audio_device": "high",     "disc_jockey": "high"},
}


# ─────────────────────────────────────────────
# 2. State schemas
# ─────────────────────────────────────────────

NOISE_OFFENSES = {
    "INVESTIGATE PROPERTY",
    "INVESTIGATE PERSON",
    "SICK ASSIST",
    "TOWED MOTOR VEHICLE",
}

NOISE_311_TYPES = {
    "Request for Snow Plowing",
    "Unshoveled Sidewalk",
    "Traffic Signal Inspection",
    "Sign Repair",
    "Recycling Cart Return",
    "Pick up Dead Animal",
}

# Maps frontend property preference labels → Boston assessment LU_DESC values
# Used in summarize to translate user selections into dataset terms for the LLM
PROPERTY_PREF_TO_LU_DESC: dict[str, list[str]] = {
    "Condo":              ["RESIDENTIAL CONDO"],
    "Single Family":      ["SINGLE FAM DWELLING"],
    "Two / Three Family": ["TWO-FAM DWELLING", "THREE-FAM DWELLING"],
    "Small Apartment":    ["APT 4-6 UNITS"],
    "Mid-Size Apartment": ["APT 7-30 UNITS"],
    "Mixed Use":          ["RES /COMMERCIAL USE", "COMM MULTI-USE"],
}


class State(TypedDict):
    # Inputs
    neighborhood:         str
    street_name:          str
    zip_code:             str
    household_type:       str | None
    property_preferences: list[str] | None
    # Parallel fetch nodes all append to this list (operator.add reducer)
    context:   Annotated[list[str],  operator.add]
    # Structured stats from fetch nodes — for UI display, not passed to LLM
    raw_stats: Annotated[list[dict], operator.add]


class OutputState(TypedDict):
    requests_311:        str
    crime_safety:        str
    property_mix:        str
    permit_activity:     str
    entertainment_scene: str
    traffic_safety:      str
    gun_violence:        str
    green_space:         str
    overall_verdict:     str
    raw_stats:           list[dict]


class NeighborhoodReport(TypedDict):
    requests_311:        str
    crime_safety:        str
    property_mix:        str
    permit_activity:     str
    entertainment_scene: str
    traffic_safety:      str
    gun_violence:        str
    green_space:         str
    overall_verdict:     str


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

        # ── context string for LLM — unchanged ──────────────────────────────
        lines = [f"Top 311 requests for {neighborhood}:\n"]
        for r in records:
            lines.append(f"  • {r['type']}: {r['count']}")

        # ── filtered stats for UI — same records, no extra API call ─────────
        meaningful = [
            {"type": r["type"], "count": int(r["count"])}
            for r in records
            if r["type"] not in NOISE_311_TYPES
        ]
        total = sum(item["count"] for item in meaningful)

        return {
            "context":   ["\n".join(lines)],
            "raw_stats": [{"section": "requests_311", "data": meaningful, "total": total}],
        }
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

        # ── context string for LLM — unchanged ──────────────────────────────
        lines = [f"Top crimes on {street_name} (District {district} – {neighborhood}, 2026):\n"]
        for r in records:
            lines.append(f"  • {r['OFFENSE_DESCRIPTION']}: {r['count']}")

        # ── filtered top 6 for UI — same records, no extra API call ───────
        filtered = [
            {"offense": r["OFFENSE_DESCRIPTION"], "count": int(r["count"])}
            for r in records
            if r["OFFENSE_DESCRIPTION"] not in NOISE_OFFENSES
        ][:6]

        return {
            "context":   ["\n".join(lines)],
            "raw_stats": [{"section": "crime_safety", "data": filtered}],
        }
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

        # ── context string for LLM — unchanged ──────────────────────────────
        lines = [f"Property mix for zip code {zip_code}:\n"]
        for r in records:
            lines.append(f"  • {r['LU_DESC']}: {r['count']}")

        # ── property counts by category for UI — same records, no extra API call ──
        lu_lookup = {r["LU_DESC"]: int(r["count"]) for r in records}
        property_counts = {
            label: sum(lu_lookup.get(lu, 0) for lu in lu_values)
            for label, lu_values in PROPERTY_PREF_TO_LU_DESC.items()
            if sum(lu_lookup.get(lu, 0) for lu in lu_values) > 0
        }

        total = sum(property_counts.values())
        return {
            "context":   ["\n".join(lines)],
            "raw_stats": [{"section": "property_mix", "data": property_counts, "total": total}],
        }
    except httpx.HTTPError as e:
        return {"context": [f"Error fetching property data: {e}"]}


async def fetch_permits(state: State) -> dict:
    """Fetch building permit activity by worktype for the zip code,
    ordered by total declared investment value, rolling 2-year window."""
    zip_code = state["zip_code"]
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
            return {"context": [f"No building permit records found for zip code: {zip_code}"]}
        lines = [f"Building permit activity for zip code {zip_code} (rolling 2-year window, ordered by total declared investment value):\n"]
        for r in records:
            worktype = r["worktype"] or "UNKNOWN"
            count = r["count"]
            total_value = float(r["total_value"] or 0)
            lines.append(f"  • {worktype}: {count} permits — ${total_value:,.0f} total declared value")
        return {"context": ["\n".join(lines)]}
    except httpx.HTTPError as e:
        return {"context": [f"Error fetching building permit data: {e}"]}


async def fetch_entertainment(state: State) -> dict:
    """Fetch entertainment license data for the zip code.
    Runs one query: entertainment type breakdown with venue and unit counts.
    """
    zip_code = state["zip_code"]

    sql_unit_types = (
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
            resp_units = await client.get(BOSTON_API_URL, params={"sql": sql_unit_types})
            resp_units.raise_for_status()

        unit_types = resp_units.json().get("result", {}).get("records", [])

        if not unit_types:
            return {"context": [f"No entertainment license records found for zip code: {zip_code}"]}

        lines = [f"Entertainment type breakdown for zip code {zip_code}:\n"]
        for r in unit_types:
            lines.append(
                f"  • {r['unit_type']}: {r['count']} venues, {r['total_units']} total units"
            )

        return {"context": ["\n".join(lines)]}

    except httpx.HTTPError as e:
        return {"context": [f"Error fetching entertainment data: {e}"]}


async def fetch_traffic_safety(state: State) -> dict:
    """Fetch traffic safety data for the street.
    Runs three queries concurrently:
      - Query A (crash mode breakdown): crash counts by mode_type since 2022
      - Query B (crash hotspots): top intersection hotspots since 2022
      - Query C (fatalities): all-time fatality records, raw list
    """
    street = state["street_name"]

    street_filter = (
        f"(\"street\" = '{street}' OR \"xstreet1\" = '{street}' OR \"xstreet2\" = '{street}')"
    )

    # Crash records — aggregated by mode_type since 2022
    sql_crash_modes = (
        'SELECT "mode_type", COUNT(*) as count '
        'FROM "e4bfe397-6bfc-49c5-9367-c879fac7401d" '
        f'WHERE {street_filter} '
        "AND \"dispatch_ts\" >= '2022-01-01' "
        'GROUP BY "mode_type" '
        'ORDER BY count DESC'
    )

    # Crash records — top hotspot intersections since 2022
    sql_crash_hotspots = (
        'SELECT "xstreet1", "xstreet2", COUNT(*) as count '
        'FROM "e4bfe397-6bfc-49c5-9367-c879fac7401d" '
        f'WHERE {street_filter} '
        "AND \"dispatch_ts\" >= '2022-01-01' "
        'GROUP BY "xstreet1", "xstreet2" '
        'ORDER BY count DESC '
        'LIMIT 5'
    )

    # Fatality records — all time, raw list
    sql_fatalities = (
        'SELECT "mode_type", "location_type", "street", "xstreet1", "xstreet2", "date_time" '
        'FROM "92f18923-d4ec-4c17-9405-4e0da63e1d6c" '
        f'WHERE {street_filter} '
        'ORDER BY "date_time" DESC'
    )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp_modes, resp_hotspots, resp_fatalities = await asyncio.gather(
                client.get(BOSTON_API_URL, params={"sql": sql_crash_modes}),
                client.get(BOSTON_API_URL, params={"sql": sql_crash_hotspots}),
                client.get(BOSTON_API_URL, params={"sql": sql_fatalities}),
            )
            resp_modes.raise_for_status()
            resp_hotspots.raise_for_status()
            resp_fatalities.raise_for_status()

        crash_modes  = resp_modes.json().get("result", {}).get("records", [])
        hotspots     = resp_hotspots.json().get("result", {}).get("records", [])
        fatalities   = resp_fatalities.json().get("result", {}).get("records", [])

        if not crash_modes and not fatalities:
            return {"context": [f"No traffic safety records found for street: {street}"]}

        lines = [f"Traffic safety data for {street}:\n"]

        # Section A — crash breakdown by mode
        if crash_modes:
            total_crashes = sum(int(r["count"]) for r in crash_modes)
            lines.append(f"Crashes since 2022 (total: {total_crashes}):")
            mode_labels = {"mv": "Motor vehicle", "ped": "Pedestrian", "bike": "Cyclist"}
            for r in crash_modes:
                label = mode_labels.get(r["mode_type"], r["mode_type"])
                lines.append(f"  • {label}: {r['count']}")
        else:
            lines.append("No crash records found since 2022.")

        lines.append("")

        # Section B — hotspot intersections
        if hotspots:
            lines.append("Most frequent crash intersections:")
            for r in hotspots:
                x1 = r["xstreet1"] or ""
                x2 = r["xstreet2"] or ""
                intersection = f"{x1} & {x2}" if x1 and x2 else x1 or x2
                lines.append(f"  • {intersection}: {r['count']} crashes")

        lines.append("")

        # Section C — fatalities
        if fatalities:
            lines.append(f"Fatalities (all time — {len(fatalities)} total):")
            mode_labels = {"mv": "Motor vehicle", "ped": "Pedestrian", "bike": "Cyclist"}
            for r in fatalities:
                label = mode_labels.get(r["mode_type"], r["mode_type"])
                x1 = r["xstreet1"] or ""
                x2 = r["xstreet2"] or ""
                intersection = f"{x1} & {x2}" if x1 and x2 else (r["street"] or "unknown location")
                date = r["date_time"][:10] if r["date_time"] else "unknown date"
                lines.append(f"  • {label} fatality — {intersection} — {date}")
        else:
            lines.append("No fatalities on record for this street.")

        return {"context": ["\n".join(lines)]}

    except httpx.HTTPError as e:
        return {"context": [f"Error fetching traffic safety data: {e}"]}


async def fetch_gun_violence(state: State) -> dict:
    """Fetch gun violence data for the neighborhood district.
    Runs two queries concurrently:
      - Query A (shootings): fatal and non-fatal shooting victims since 2022
      - Query B (shots fired): total shots fired and ballistics-confirmed incidents since 2022
    """
    neighborhood = state["neighborhood"]
    district = NEIGHBORHOOD_TO_DISTRICT.get(neighborhood)
    if not district:
        return {"context": [f"Could not resolve a BPD district for neighborhood: '{neighborhood}'."]} 

    sql_shootings = (
        'SELECT "shooting_type_v2", COUNT(*) as count '
        'FROM "73c7e069-701f-4910-986d-b950f46c91a1" '
        f"WHERE \"district\" = '{district}' "
        "AND \"shooting_date\" >= '2022-01-01' "
        'GROUP BY "shooting_type_v2" '
        'ORDER BY count DESC'
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
            resp_shootings.raise_for_status()
            resp_shots.raise_for_status()

        shootings  = resp_shootings.json().get("result", {}).get("records", [])
        shots_data = resp_shots.json().get("result", {}).get("records", [])

        lines = [f"Gun violence data for {neighborhood} (District {district}) since 2022:\n"]

        # Section A — shooting victims
        if shootings:
            total_victims = sum(int(r["count"]) for r in shootings)
            lines.append(f"Shooting victims (total: {total_victims}):")
            for r in shootings:
                lines.append(f"  • {r['shooting_type_v2']}: {r['count']}")
        else:
            lines.append("No shooting victim records found since 2022.")

        lines.append("")

        # Section B — shots fired
        if shots_data:
            r = shots_data[0]
            total  = r["total_shots_fired"] or 0
            confirmed = r["confirmed_with_ballistics"] or 0
            lines.append(f"Shots fired incidents: {total} total, {confirmed} confirmed with ballistics evidence")
        else:
            lines.append("No shots fired records found since 2022.")

        return {"context": ["\n".join(lines)]}

    except httpx.HTTPError as e:
        return {"context": [f"Error fetching gun violence data: {e}"]}


async def fetch_green_space(state: State) -> dict:
    """Fetch green space data for the neighborhood.
    Runs three queries concurrently:
      - Query A (trees): total street tree count via ArcGIS
      - Query B (open space breakdown): open space types and acreage via CKAN
      - Query C (recreational acres): total usable acres excluding cemeteries via CKAN
    """
    neighborhood = state["neighborhood"]
    bprd_neighborhood = NEIGHBORHOOD_TO_BPRD.get(neighborhood, neighborhood)

    arcgis_url = (
        "https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/"
        "BPRD_Trees/FeatureServer/0/query"
    )

    open_space_breakdown_sql = (
        'SELECT "TypeLong", COUNT(*) as count, '
        'SUM(CAST("ACRES" AS NUMERIC)) as total_acres '
        'FROM "61c0239f-c8fd-47de-8375-2405382ef37c" '
        f"WHERE \"DISTRICT\" = '{neighborhood}' "
        'GROUP BY "TypeLong" '
        'ORDER BY total_acres DESC'
    )

    recreational_acres_sql = (
        'SELECT SUM(CAST("ACRES" AS NUMERIC)) as recreational_acres '
        'FROM "61c0239f-c8fd-47de-8375-2405382ef37c" '
        f"WHERE \"DISTRICT\" = '{neighborhood}' "
        "AND \"TypeLong\" != 'Cemeteries & Burying Grounds'"
    )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp_trees, resp_breakdown, resp_rec = await asyncio.gather(
                client.get(arcgis_url, params={
                    "where": f"neighborhood='{bprd_neighborhood}'",
                    "returnCountOnly": "true",
                    "f": "json",
                }),
                client.get(BOSTON_API_URL, params={"sql": open_space_breakdown_sql}),
                client.get(BOSTON_API_URL, params={"sql": recreational_acres_sql}),
            )
            resp_trees.raise_for_status()
            resp_breakdown.raise_for_status()
            resp_rec.raise_for_status()

        tree_count    = resp_trees.json().get("count", 0)
        os_breakdown  = resp_breakdown.json().get("result", {}).get("records", [])
        rec_data      = resp_rec.json().get("result", {}).get("records", [])

        lines = [f"Green space data for {neighborhood}:\n"]

        # Section A — tree count
        lines.append(f"Total street and park trees: {tree_count}")
        lines.append("")

        # Section B — open space breakdown
        if os_breakdown:
            lines.append("Open space by type:")
            for r in os_breakdown:
                acres = float(r["total_acres"] or 0)
                lines.append(
                    f"  • {r['TypeLong']}: {r['count']} spaces — {acres:.1f} acres"
                )
        else:
            lines.append("No open space records found for this neighborhood.")

        lines.append("")

        # Section C — recreational acres
        if rec_data and rec_data[0]["recreational_acres"]:
            rec_acres = float(rec_data[0]["recreational_acres"])
            lines.append(f"Total usable recreational acres (excluding cemeteries): {rec_acres:.1f}")
        else:
            lines.append("No recreational acre data found.")

        return {"context": ["\n".join(lines)]}

    except httpx.HTTPError as e:
        return {"context": [f"Error fetching green space data: {e}"]}


# ─────────────────────────────────────────────
# 5. Async summarization node
# ─────────────────────────────────────────────

sys_msg = SystemMessage(content=(
    "You are a straight-talking neighborhood-analysis assistant for Boston house hunters. "
    "Your job is to give honest, specific, buyer-focused insights — not reassuring generalities. "
    "A buyer making a major financial decision needs the full picture, including red flags. "
    "You have deep knowledge of Boston's neighborhoods from your training data — use it. "
    "Where the raw data alone is thin or ambiguous, draw on what you know about how this neighborhood "
    "has been evolving in the 2020s: gentrification pressures, demographic shifts, development pipelines, "
    "school quality, commute patterns, and community character. Blend data and knowledge into a richer picture.\n\n"
    "You will receive eight raw data blocks (311 complaints, crime incidents, property mix, "
    "building permits, entertainment licenses, traffic safety, gun violence, green space). "
    "Produce a structured report with exactly nine fields as defined. "
    "Each field should be substantive — aim for 3-5 sentences minimum per section, "
    "combining the numbers from the data with what you know about this neighborhood.\n\n"
    "## 311 Service Requests\n"
    "  - 'CE Collection' = Code Enforcement Collection: the city is actively pursuing a landlord or "
    "property owner to fix a code violation. High counts mean properties on this street or in this "
    "neighborhood have unresolved maintenance failures that required city enforcement — a direct red flag "
    "for renters evaluating landlord quality.\n"
    "  - 'Needle Pickup' and 'Encampments' = visible substance-abuse or homelessness activity on public streets — state plainly.\n"
    "  - 'Unsatisfactory Living Conditions' and 'Heat - Excessive/Insufficient' = tenants reporting "
    "uninhabitable conditions or landlord failure to provide heat — serious housing quality flags.\n"
    "  - 'Space Savers' = residents reserving public parking spots after snow — signals acute parking scarcity.\n"
    "  - High 'Illegal Dumping' or 'Abandoned Vehicles' = signs of neighborhood neglect or deferred maintenance.\n"
    "  - 'Parking Enforcement' at high volume = active enforcement pressure on a dense street.\n"
    "  - Use your knowledge of this neighborhood to contextualize these numbers — "
    "a high CE Collection count in a rapidly gentrifying area means something different than in a "
    "long-neglected one. Say what it means for someone moving here today.\n\n"
    "## Crime & Safety\n"
    "  - Context matters: commercial streets have higher expected counts than residential ones. "
    "Calibrate your interpretation accordingly and name the street type.\n"
    "  - Distinguish procedural incidents (Investigate Person/Property) from actual crimes — "
    "procedural entries reflect police activity, not necessarily danger.\n"
    "  - Flag auto theft, drug offenses, robbery, assault, and threats explicitly — do not minimize.\n"
    "  - Low counts on a quiet residential street are genuinely reassuring — say so directly.\n"
    "  - If fewer than 10 total incidents are found, describe what was found, then add one brief sentence "
    "at the end noting that early-year data may be incomplete. Never lead with the caveat.\n"
    "  - Use your knowledge of this neighborhood's safety reputation in the 2020s to add context — "
    "has it improved, stayed flat, or worsened? Are there specific blocks or intersections known for issues?\n"
    "  - Never mention sparse data, data limitations, or data quality in the overall_verdict.\n\n"
    "## Property Mix\n"
    "  - 'CONDO PARKING (RES)' = parking sold separately — a meaningful extra cost in Boston's market.\n"
    "  - 'SUBSD HOUSING S-8' = Section 8 subsidized housing — note count and what it signals about tenure mix.\n"
    "  - 'CITY OF BOSTON' / 'BOST REDEVELOP AUTH' / 'BOS HOUSING AUTHOR' = publicly owned land — "
    "could signal long-term affordable housing or future development site.\n"
    "  - 'RES LAND (Unusable)' = vacant or blighted lots — potential development pressure or blight.\n"
    "  - Two- and three-family home dominance = rental-heavy neighborhood with owner-occupier landlords.\n"
    "  - If property preferences are provided, state explicitly whether strong or weak inventory exists "
    "for those types — cite the actual count. Do not give a generic answer when numbers are available.\n"
    "  - Use your knowledge of this zip code's housing market in the 2020s to contextualize — "
    "is condo inventory growing fast? Are three-families being converted? What does the mix signal about "
    "who lives here and who is moving in?\n\n"
    "## Building Permits\n"
    "  - Data covers the rolling past two years, ordered by total declared investment value.\n"
    "  - Never use raw worktype codes. Describe what is happening in plain English: new buildings going up, "
    "buildings being converted between uses, gut renovations, interior upgrades, and so on. "
    "Cite dollar values and permit counts for the top 3 categories.\n"
    "  - Open by stating the neighborhood's approximate geographic size and calibrating the numbers against it — "
    "the reader needs to know whether this is concentrated activity they will encounter daily "
    "or investment spread across a large area.\n"
    "  - Write for renters first, like a friend warning someone before they sign a lease. "
    "Every insight should land on something concrete: what to expect at renewal, whether their building type "
    "is at risk of conversion, whether new supply will help or hurt their rent. "
    "High new construction — say whether units are likely luxury or mid-market. "
    "High conversions — flag displacement risk. "
    "High renovations — say whether landlords are retaining tenants or justifying increases. "
    "Low activity — note it as a stability signal.\n"
    "  - For buyers, include a line or two in the middle on appreciation potential. "
    "Never close with the buyer note.\n"
    "  - Close by connecting permits to at least one other dataset — "
    "311, crime, green space, or traffic — to show the combined picture.\n"
    "  - Draw on your knowledge of this neighborhood's development trajectory in the 2020s: "
    "mid-transformation or established, and what this investment signals for the next 3-5 years.\n\n"
    "## Entertainment Scene\n"
    "  - You will receive raw entertainment license data with unit types, venue counts, and total units. "
    "This is your primary input — analyze what the mix of licenses tells you about the neighborhood. "
    "You will also receive a pre-computed entertainment profile that ranks this zip code's casual dining, "
    "bar, and nightlife density as high, moderate, or low relative to all Boston neighborhoods. "
    "Use this to contextualize the raw numbers — it tells you whether the counts you are looking at "
    "are typical, above average, or sparse for Boston. "
    "Do not quote license terms or numbers directly. Describe the neighborhood's entertainment personality.\n"
    "  - The data is a proxy for what the neighborhood feels like in the evening and on weekends. "
    "A high count of screen and audio-based licenses means a bar and restaurant scene built around "
    "casual dining and sports TV. A high count of DJ, dancing, and live music licenses means active "
    "nightlife with real noise. Stage and performance licenses mean arts and cultural programming. "
    "Read the mix and describe the vibe — don't inventory the license types.\n"
    "  - Anchor your description in what you know from your training data about the actual bars, "
    "restaurants, venues, and cultural institutions in this neighborhood. Name the kind of establishments "
    "that define the area, whether it draws people from other neighborhoods or serves locals, "
    "and what a typical Friday night looks and sounds like. The license data should confirm or add "
    "nuance to that picture, not replace it.\n"
    "  - If the data shows significant nightlife or live music presence, say plainly what that means "
    "for noise on residential streets — especially for someone deciding whether to sign a lease nearby.\n"
    "  - If a buyer profile is provided, say whether this entertainment scene is a fit or a friction "
    "point for that household type — a couple without kids has different tolerance than a family "
    "with young children.\n\n"
    "## Traffic Safety\n"
    "  - You will receive three sub-blocks: crash mode breakdown, hotspot intersections, and fatalities.\n"
    "  - Always state total crash count since 2022 and break it down by mode — motor vehicle, pedestrian, cyclist.\n"
    "  - Pedestrian and cyclist crashes are the most buyer-relevant — flag these explicitly with counts. "
    "A high pedestrian crash count on a residential street is a genuine red flag.\n"
    "  - Name specific hotspot intersections and their crash counts — do not summarize vaguely.\n"
    "  - Any fatality must be flagged explicitly: mode type, intersection, date. "
    "Zero fatalities is a meaningful positive — say so.\n"
    "  - Commercial and arterial streets naturally see more crashes than quiet residential ones — calibrate.\n"
    "  - Use your knowledge of this street and neighborhood's traffic conditions in the 2020s — "
    "is this a known dangerous corridor? Have there been Vision Zero interventions here? "
    "Does the data confirm or contradict the neighborhood's reputation?\n\n"
    "## Gun Violence\n"
    "  - You will receive two sub-blocks: shooting victims and shots fired incidents since 2022.\n"
    "  - Always state total shooting victims and break down by Fatal vs Non-Fatal — cite exact numbers.\n"
    "  - Fatal shootings are the highest severity signal — flag any fatals explicitly with counts.\n"
    "  - Zero fatal shootings is a meaningful positive — state it clearly.\n"
    "  - Shots fired incidents signal active gun presence in the district even when victim counts are low.\n"
    "  - This data is at the district level, not the street level — a district covers multiple neighborhoods. "
    "Calibrate accordingly and note this context for the buyer.\n"
    "  - If crime data shows low incidents on the specific street but district gun violence is high, "
    "flag the discrepancy — the street may be calm but the surrounding area is not.\n"
    "  - Use your knowledge of this district's gun violence trajectory in the 2020s — "
    "is it improving, worsening, or concentrated in specific pockets? Help the buyer understand "
    "whether these numbers are typical for this area or anomalous.\n"
    "  - Never soften or hedge — a buyer has the right to know this plainly.\n\n"
    "## Green Space\n"
    "  - You will receive three sub-blocks: total tree count, open space breakdown by type, and total recreational acres.\n"
    "  - Always state the total tree count — high numbers signal a green, well-canopied neighborhood.\n"
    "  - Reference the open space breakdown by type and cite total recreational acres excluding cemeteries.\n"
    "  - 'Parks, Playgrounds & Athletic Fields' = most buyer-relevant category — cite count and acreage.\n"
    "  - 'Urban Wilds' = natural woodland, high value for nature-oriented buyers.\n"
    "  - 'Community Gardens' = signals an engaged, sustainability-minded community.\n"
    "  - Cemeteries = open land but not usable recreational space — exclude from livability count.\n"
    "  - If a buyer profile is provided, tailor the green space analysis: families need playgrounds, "
    "retirees value walkable parks, investors care less.\n"
    "  - Use your knowledge of this neighborhood's parks and green corridors — are there well-known "
    "parks nearby? Is the neighborhood walkable? Does the tree count reflect the street-level experience?\n\n"
    "## Buyer Profile & Property Preference\n"
    "  - If a buyer profile (household type) is provided, weave it throughout the analysis — "
    "not just in property_mix and overall_verdict, but wherever the data has implications for that buyer type.\n"
    "  - If property preferences are provided, always state in property_mix whether the neighborhood has "
    "strong or weak inventory for those specific types — cite the actual count from the data.\n"
    "  - In overall_verdict, name the buyer type and property preference explicitly. "
    "Do not give a generic verdict when this context is available.\n\n"
    "## Overall Verdict\n"
    "Synthesize across all eight datasets into a comprehensive, substantive verdict of at least 250 words. "
    "Structure it as follows:\n"
    "1. Open with a concise characterization of the neighborhood's identity and everyday character — "
    "its dominant vibe, the kind of people and energy you encounter on the street, "
    "and how it distinguishes itself from adjacent Boston neighborhoods. "
    "Ground this in what the data reveals (entertainment mix signals nightlife culture, "
    "311 patterns signal street-level quality, property mix signals tenure and demographic makeup, "
    "green space signals walkability and family-friendliness) but enrich it with what you know about "
    "the neighborhood's reputation and evolution in the 2020s. "
    "Then pivot to a direct assessment of whether that character fits the buyer's profile and preferences — "
    "not a generic trajectory label, but a genuine answer to 'is this the right place for me?' "
    "If no buyer profile is provided, characterize who the neighborhood is currently attracting "
    "and who it may not suit.\n"
    "2. Connect the most important signals across datasets — how permit activity relates to 311 complaints, "
    "how traffic safety relates to crime patterns, how entertainment density relates to property mix and lifestyle fit.\n"
    "3. Name specific red flags a buyer must know before committing — be explicit with numbers, not vague.\n"
    "4. Name specific green flags that support buying — be specific with numbers and neighborhood context.\n"
    "5. Close with a direct, preference-aware recommendation: given this buyer's household type and property "
    "preferences, should they buy here, look elsewhere, or proceed with specific conditions? "
    "Name what would have to be true for this to be the right move. "
    "Be direct — a buyer making a major financial decision deserves a real opinion, not hedged generalities."
))


async def summarize(state: State) -> dict:
    """Call the LLM with all fetched context and return the nine structured fields."""
    household = state.get("household_type")
    prefs     = state.get("property_preferences")
    buyer_line = f"Buyer profile: {household}\n" if household else ""

    # Translate frontend labels to dataset LU_DESC terms so the LLM
    # sees the exact property type strings present in the context block
    if prefs:
        translated = []
        for p in prefs:
            lu_values = PROPERTY_PREF_TO_LU_DESC.get(p)
            if lu_values:
                translated.append(f"{p} ({', '.join(lu_values)})")
            else:
                translated.append(p)
        pref_line = f"Property preference: {', '.join(translated)}\n"
    else:
        pref_line = ""

    # Build entertainment profile line from pre-computed tiers
    zip_info = ZIP_CODE_INFO.get(state["zip_code"], {})
    if zip_info:
        entertainment_line = (
            f"Entertainment profile for {zip_info['name']} ({state['zip_code']}): "
            f"casual dining/bar scene: {zip_info.get('widescreen_tv', 'unknown')}, "
            f"background music scene: {zip_info.get('audio_device', 'unknown')}, "
            f"nightlife/DJ scene: {zip_info.get('disc_jockey', 'unknown')}\n"
        )
    else:
        entertainment_line = ""

    user_msg = HumanMessage(content=(
        f"Neighborhood: {state['neighborhood']}\n"
        f"Street: {state['street_name']}\n"
        f"Zip Code: {state['zip_code']}\n"
        + buyer_line
        + pref_line
        + entertainment_line
        + "\n"
        + "\n\n".join(state["context"])
    ))
    report = await llm_with_structure.ainvoke([sys_msg, user_msg])
    return {
        "requests_311":        report["requests_311"],
        "crime_safety":        report["crime_safety"],
        "property_mix":        report["property_mix"],
        "permit_activity":     report["permit_activity"],
        "entertainment_scene": report["entertainment_scene"],
        "traffic_safety":      report["traffic_safety"],
        "gun_violence":        report["gun_violence"],
        "green_space":         report["green_space"],
        "overall_verdict":     report["overall_verdict"],
    }


# ─────────────────────────────────────────────
# 6. Build the parallelized graph
# ─────────────────────────────────────────────

builder = StateGraph(State, output_schema=OutputState)

builder.add_node("fetch_311",            fetch_311)
builder.add_node("fetch_crime",          fetch_crime)
builder.add_node("fetch_property",       fetch_property)
builder.add_node("fetch_permits",        fetch_permits)
builder.add_node("fetch_entertainment",  fetch_entertainment)
builder.add_node("fetch_traffic_safety", fetch_traffic_safety)
builder.add_node("fetch_gun_violence",   fetch_gun_violence)
builder.add_node("fetch_green_space",    fetch_green_space)
builder.add_node("summarize",            summarize)

builder.add_edge(START,                 "fetch_311")
builder.add_edge(START,                 "fetch_crime")
builder.add_edge(START,                 "fetch_property")
builder.add_edge(START,                 "fetch_permits")
builder.add_edge(START,                 "fetch_entertainment")
builder.add_edge(START,                 "fetch_traffic_safety")
builder.add_edge(START,                 "fetch_gun_violence")
builder.add_edge(START,                 "fetch_green_space")

builder.add_edge("fetch_311",            "summarize")
builder.add_edge("fetch_crime",          "summarize")
builder.add_edge("fetch_property",       "summarize")
builder.add_edge("fetch_permits",        "summarize")
builder.add_edge("fetch_entertainment",  "summarize")
builder.add_edge("fetch_traffic_safety", "summarize")
builder.add_edge("fetch_gun_violence",   "summarize")
builder.add_edge("fetch_green_space",    "summarize")

builder.add_edge("summarize", END)

graph = builder.compile()
