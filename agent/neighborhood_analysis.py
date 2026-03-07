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
    requests_311:        str
    crime_safety:        str
    property_mix:        str
    permit_activity:     str
    entertainment_scene: str
    traffic_safety:      str
    gun_violence:        str
    green_space:         str
    overall_verdict:     str


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
    "A buyer making a major financial decision needs the full picture, including red flags.\n\n"
    "You will receive eight raw data blocks (311 complaints, crime incidents, property mix, "
    "building permits, entertainment licenses, traffic safety, gun violence, green space). "
    "Produce a structured report with exactly nine fields as defined.\n\n"
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
    "  - Low counts on a quiet residential street are reassuring — say so.\n"
    "  - If fewer than 10 total incidents are found, describe what was found, then add one brief sentence \n"
    "at the end noting that early-year data may be incomplete. Never lead with the caveat.\n"
    "  - Never mention sparse data, data limitations, or data quality in the overall_verdict — \n"
    "that field is for buyer insights only.\n\n"
    "## Property Mix\n"
    "  - 'CONDO PARKING (RES)' = parking sold separately — extra cost for buyers.\n"
    "  - 'SUBSD HOUSING S-8' = Section 8 subsidized housing.\n"
    "  - 'CITY OF BOSTON' / 'BOST REDEVELOP AUTH' / 'BOS HOUSING AUTHOR' = publicly owned land.\n"
    "  - 'RES LAND (Unusable)' = vacant lots — possible blight or future development.\n"
    "  - Two- and three-family home dominance = rental-heavy neighborhood.\n\n"
    "## Building Permits\n"
    "  - Data covers the rolling past two years. Results are ordered by total declared "
    "investment value, not count — so the top entries represent the highest financial "
    "impact activity, not just the most frequent.\n"
    "  - Always reference specific dollar values and counts for the top 3 worktypes — "
    "do not summarize without numbers. Example: 'ERECT: 68 permits totalling $430M signals large-scale development'.\n"
    "  - 'ERECT' = new construction. High total value here means large developments, not "
    "just small single-family builds. Flag explicitly — rising values likely but "
    "neighborhood character is actively shifting.\n"
    "  - 'COB' = Certificate of Occupancy Boston. Large completed projects receiving final "
    "city sign-off. High value means major buildings recently finishing construction.\n"
    "  - 'NROCC' = New occupancy permit. Similar to COB — signals recently completed "
    "developments entering the market.\n"
    "  - 'CHGOCC' = Change of occupancy. Buildings converting from one use type to another "
    "— single family to multi-family, commercial to residential, or vice versa. "
    "High counts signal a neighborhood under active conversion pressure.\n"
    "  - 'INTEXT' / 'INTREN' / 'EXTREN' = renovation of existing stock. High combined "
    "value means owners are actively maintaining and upgrading — healthy signal.\n"
    "  - 'INTDEM' = interior demolition. Precedes gut renovations or condo conversions. "
    "Flag if high — signals transformation of existing residential stock.\n"
    "  - High 'SOL' + 'INSUL' total value = owner-occupier culture investing in long-term "
    "efficiency. Positive signal for neighborhood stability.\n"
    "  - Connect to other datasets: high ERECT + CHGOCC alongside rising CE Collection "
    "in 311 = development-driven displacement pressure. High ERECT with low crime "
    "and low needle pickup = growth without deterioration — strong buy signal.\n\n"
    "## Entertainment Scene\n"
    "  - You will receive an entertainment type breakdown showing unit types, venue counts, and total units.\n"
    "  - Dominant unit types reveal neighborhood character: "
    "Widescreen TV / Audio Device / Radio / Cassette = casual bar and restaurant scene, low noise impact. "
    "Disc Jockey / Dancing by Patrons / Instrument / Vocal = active nightlife, higher noise impact. "
    "Stage Play / Floor Show / Karaoke = arts and performance culture.\n"
    "  - High counts of Disc Jockey, Dancing by Patrons, or Night Club types = flag for buyers prioritizing quiet.\n"
    "  - Low counts dominated by TVs and audio devices = relaxed, low-impact entertainment scene — reassuring for families.\n"
    "  - Always cite the top 3 unit types with their venue counts — do not summarize without numbers.\n\n"
    "## Traffic Safety\n"
    "  - You will receive three sub-blocks: crash mode breakdown, hotspot intersections, and fatalities.\n"
    "  - Always state total crash count since 2022 and break it down by mode — motor vehicle, pedestrian, cyclist.\n"
    "  - Pedestrian and cyclist crashes are the most buyer-relevant — flag these explicitly. "
    "A high pedestrian crash count on a residential street is a red flag.\n"
    "  - Name the specific hotspot intersections and their crash counts — do not summarize vaguely.\n"
    "  - Fatalities are the highest severity signal. Any fatality on the street must be flagged "
    "explicitly with mode type, intersection, and date. Zero fatalities is a meaningful positive — say so.\n"
    "  - Context matters: commercial and arterial streets will naturally have more crashes than "
    "quiet residential streets. Calibrate accordingly.\n"
    "  - Connect to other data: high pedestrian crashes alongside high Needle Pickup in 311 "
    "and low streetlight presence = compounding walkability risk. "
    "Zero crashes and zero fatalities on a quiet street = strong safety signal for families.\n\n"
    "## Gun Violence\n"
    "  - You will receive two sub-blocks: shooting victims and shots fired incidents since 2022.\n"
    "  - Always state total shooting victims and break down by Fatal vs Non-Fatal — cite exact numbers.\n"
    "  - Fatal shootings are the highest severity signal — flag any fatals explicitly.\n"
    "  - Zero fatal shootings is a meaningful positive — state it clearly.\n"
    "  - Shots fired incidents (even without victims) signal active gun presence in the district — \n"
    "    high counts should be flagged even if shootings are low.\n"
    "  - Context matters: this data is at the district level, not the street level. \n"
    "    A district covers multiple neighborhoods — calibrate accordingly and note this for the buyer.\n"
    "  - Connect to crime data: if crime tool shows low incidents but gun violence is high, \n"
    "    flag the discrepancy — the street may be quiet but the surrounding district is not.\n"
    "  - Never soften or hedge gun violence data — a buyer has the right to know this plainly.\n\n"
    "## Green Space\n"
    "  - You will receive three sub-blocks: total tree count, open space breakdown by type, and total recreational acres.\n"
    "  - Always state the total tree count — high numbers signal a green, well-canopied neighborhood.\n"
    "  - Reference the open space breakdown by type and cite total recreational acres (excluding cemeteries).\n"
    "  - 'Parks, Playgrounds & Athletic Fields' = most buyer-relevant category — cite count and acreage.\n"
    "  - 'Urban Wilds' = natural woodland areas, high value for nature-oriented buyers.\n"
    "  - 'Community Gardens' = signals an engaged, sustainability-minded community.\n"
    "  - 'Cemeteries & Burying Grounds' = open land but not usable recreational space — do not count toward green livability.\n"
    "  - Connect to buyer preferences: families with children should have park and playground acreage highlighted. \n"
    "  - High tree count + high recreational acres = strong livability signal. \n"
    "  - Low tree count + minimal open space = neighborhood lacks green infrastructure — flag for buyers who value outdoor access.\n\n"
    "## Overall Verdict\n"
    "Synthesize across all eight datasets into a comprehensive, substantive verdict of at least 150 words. "
    "Structure it as follows:\n"
    "1. Open with a one-sentence characterization of the neighborhood trajectory — stable, early transformation, or mid-gentrification.\n"
    "2. Connect the most important signals across datasets — e.g. how permit activity relates to 311 complaints, "
    "how traffic safety relates to crime, how entertainment density relates to property mix.\n"
    "3. Name specific red flags a buyer must know before purchasing — be explicit, not vague.\n"
    "4. Name specific green flags that support buying — again, be specific with numbers where possible.\n"
    "5. Close with a direct recommendation tailored to buyer type: "
    "who this neighborhood is right for and who should look elsewhere. "
    "Be direct. A buyer making a major financial decision deserves a real opinion, not hedged generalities."
))


async def summarize(state: State) -> dict:
    """Call the LLM with all fetched context and return the nine structured fields."""
    user_msg = HumanMessage(content=(
        f"Neighborhood: {state['neighborhood']}\n"
        f"Street: {state['street_name']}\n"
        f"Zip Code: {state['zip_code']}\n\n"
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
