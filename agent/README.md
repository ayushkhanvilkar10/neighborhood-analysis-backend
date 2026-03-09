# Neighborhood Analysis Agent

A LangGraph-powered agent that analyzes Boston neighborhoods by making eight parallel calls to the Boston Open Data API and the ArcGIS REST API, then synthesizing the results into a structured report using GPT-4o. Supports optional buyer personalization via household type and property preferences.

---

## Architecture

```
START
  ├── fetch_311             ─┐
  ├── fetch_crime           ─┤
  ├── fetch_property        ─┤
  ├── fetch_permits         ─┼── (parallel) ──► summarize ──► END
  ├── fetch_entertainment   ─┤
  ├── fetch_traffic_safety  ─┤
  ├── fetch_gun_violence    ─┤
  └── fetch_green_space     ─┘
```

All eight fetch nodes run in parallel using LangGraph's fan-out/fan-in pattern. Each node writes its result to a shared `context` list in state using an `operator.add` reducer. Once all eight finish, LangGraph fans back in to the `summarize` node, which passes the full context to GPT-4o and returns a validated nine-field report.

---

## Graph State

```python
class State(TypedDict):
    neighborhood:         str         # e.g. "Jamaica Plain"
    street_name:          str         # e.g. "CREIGHTON ST" (uppercase, as in BPD dataset)
    zip_code:             str         # e.g. "02130"
    household_type:       str | None  # e.g. "partner", "family", "single" — optional
    property_preferences: list[str] | None  # e.g. ["Condo"] — max 2, optional
    context: Annotated[list[str], operator.add]  # populated by the eight fetch nodes
```

**Output:**

```python
class OutputState(TypedDict):
    requests_311:        str   # Analysis of 311 service request patterns
    crime_safety:        str   # Analysis of crime incidents on the street
    property_mix:        str   # Analysis of property types in the zip code
    permit_activity:     str   # Analysis of building permit investment activity
    entertainment_scene: str   # Analysis of entertainment license unit types
    traffic_safety:      str   # Analysis of crashes and fatalities on the street
    gun_violence:        str   # Analysis of shootings and shots fired in the district
    green_space:         str   # Analysis of trees and open space in the neighborhood
    overall_verdict:     str   # Synthesized buyer-focused verdict across all datasets
```

---

## Buyer Personalization

When `household_type` and/or `property_preferences` are provided, the `summarize` node injects them into the human message sent to GPT-4o:

```
Buyer profile: partner
Property preference: Condo
```

The system prompt then instructs the LLM to:
- Address the buyer directly by household type in both `property_mix` and `overall_verdict`
- State explicitly whether the neighborhood has strong or weak inventory for the preferred property types, citing actual counts from the data
- Never produce a generic verdict when buyer context is available

**Supported `household_type` values:**

| Value | Label |
|---|---|
| `single` | Living solo |
| `partner` | Couple / Partner |
| `family` | Family with kids |
| `retiree` | Retiree / Empty nester |
| `investor` | Investor |

**Supported `property_preferences` values** (max 2):

`Condo`, `Single Family`, `Multi-Family`, `Townhouse`, `New Construction`, `Fixer-Upper`

---

## Neighborhood Mappings

Two lookup dictionaries are defined at the top of `neighborhood_analysis.py`. Both use the same 21 canonical neighborhood keys.

### `NEIGHBORHOOD_TO_DISTRICT`
Maps neighborhood names to BPD district codes. Used by `fetch_crime` and `fetch_gun_violence`.

| Neighborhood | District |
|---|---|
| Back Bay / South End / Fenway | D4 |
| Beacon Hill / Downtown | A1 |
| Charlestown | A15 |
| East Boston | A7 |
| Roxbury | B2 |
| Mattapan / Greater Mattapan | B3 |
| Dorchester | C11 |
| South Boston | C6 |
| Jamaica Plain / Mission Hill / Roslindale | E13 |
| West Roxbury | E5 |
| Hyde Park | E18 |
| Allston / Brighton | D14 |

### `NEIGHBORHOOD_TO_BPRD`
Maps neighborhood names to BPRD Trees ArcGIS neighborhood values. Used by `fetch_green_space`.

| Agent Neighborhood | BPRD Value |
|---|---|
| Back Bay / Beacon Hill | `Back Bay/Beacon Hill` |
| Allston / Brighton | `Allston-Brighton` |
| Fenway / Kenmore / Audubon Circle / Longwood | `Fenway/Longwood` |
| Downtown / Financial District | `Central Boston` |
| All others | Direct match |

---

## Data Sources

All CKAN sources are queried via:
```
https://data.boston.gov/api/3/action/datastore_search_sql?sql=<query>
```

The BPRD Trees source is queried via the ArcGIS REST API:
```
https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/BPRD_Trees/FeatureServer/0/query
```

All HTTP calls are made asynchronously using `httpx.AsyncClient` with a 15-second timeout.

---

### Tool 1 — `fetch_311` (311 Service Requests)

Fetches the top 15 most frequent 311 complaint types for the neighborhood.

**Dataset ID:** `1a0b420d-99f1-4887-9851-990b2a5a6e17`
**Filter:** `neighborhood`

**Signal types flagged in the prompt:**
- `CE Collection` → Code Enforcement (city pursuing code violations)
- `Needle Pickup` / `Encampments` → visible substance abuse or homelessness
- `Unsatisfactory Living Conditions` / `Heat - Excessive/Insufficient` → housing quality issues
- `Space Savers` → parking scarcity post-snowstorm
- `Illegal Dumping` / `Abandoned Vehicles` → neighborhood neglect

---

### Tool 2 — `fetch_crime` (Crime & Safety)

Fetches the top 15 most frequent crime types on the specific street in the current year.

**Dataset ID:** `b973d8cb-eeb2-4e7e-99da-c92938efc9c0`
**Filter:** `DISTRICT` (resolved via `NEIGHBORHOOD_TO_DISTRICT`) + `STREET` + `YEAR = '2026'`

**Signal types flagged in the prompt:**
- `INVESTIGATE PERSON` / `INVESTIGATE PROPERTY` → procedural, not actual crimes
- Auto theft, drug offenses, robbery, threats → flagged explicitly
- Low counts on a residential street → noted as reassuring
- If fewer than 10 incidents, data limitation noted briefly at end of field

---

### Tool 3 — `fetch_property` (Property Assessment)

Fetches the top 15 most frequent property use types in the zip code.

**Dataset ID:** `ee73430d-96c0-423e-ad21-c4cfb54c8961`
**Filter:** `ZIP_CODE`

**Signal types flagged in the prompt:**
- `CONDO PARKING (RES)` → parking sold separately, extra cost
- `SUBSD HOUSING S-8` → Section 8 subsidized housing
- `CITY OF BOSTON` / `BOST REDEVELOP AUTH` → publicly owned land
- `RES LAND (Unusable)` → vacant lots, blight or future development
- Two- and three-family home dominance → rental-heavy neighborhood

---

### Tool 4 — `fetch_permits` (Building Permits)

Fetches building permit activity by worktype, ordered by total declared investment value, over a rolling 2-year window.

**Dataset ID:** `6ddcd912-32a0-43df-9908-63574f8c7e77`
**Filter:** `zip` + `issued_date >= CURRENT_DATE - INTERVAL '2 years'`

**Worktype codes flagged in the prompt:**
- `ERECT` → new construction
- `COB` → Certificate of Occupancy (completed projects)
- `NROCC` → new occupancy permit
- `CHGOCC` → change of occupancy (conversion pressure)
- `INTEXT` / `INTREN` / `EXTREN` → renovation of existing stock
- `INTDEM` → interior demolition (precedes gut renovation)
- `SOL` / `INSUL` → sustainability investment, owner-occupier signal

---

### Tool 5 — `fetch_entertainment` (Entertainment Licenses)

Fetches entertainment unit type breakdown for the zip code — showing how many venues have each entertainment type and total units.

**Dataset ID:** `1c4c1f7c-9a2a-4f4f-85a7-d3462c6bc9cb`
**Filter:** `zip` + `status = 'Active'`

**Interpretation guidance in the prompt:**
- `Widescreen TV` / `Audio Device` / `Radio` → casual bar and restaurant scene, low noise
- `Disc Jockey` / `Dancing by Patrons` / `Instrument` / `Vocal` → active nightlife, higher noise
- `Stage Play` / `Floor Show` / `Karaoke` → arts and performance culture
- High DJ / Dancing counts → flagged for noise-sensitive buyers

---

### Tool 6 — `fetch_traffic_safety` (Vision Zero Crashes + Fatalities)

Runs three concurrent queries for the specific street:
- Crash counts by mode type (`mv`, `ped`, `bike`) since 2022
- Top 5 hotspot intersections by crash frequency since 2022
- All-time fatality records with mode, intersection, and date

**Crash dataset ID:** `e4bfe397-6bfc-49c5-9367-c879fac7401d`
**Fatality dataset ID:** `92f18923-d4ec-4c17-9405-4e0da63e1d6c`
**Filter:** `street OR xstreet1 OR xstreet2 = street_name`

**Key signals:**
- Pedestrian and cyclist crashes → most buyer-relevant, flagged explicitly
- Named hotspot intersections → cited with crash counts
- Any fatality → flagged with mode, location, and date
- Zero fatalities → stated as a meaningful positive

---

### Tool 7 — `fetch_gun_violence` (Shootings + Shots Fired)

Runs two concurrent queries for the BPD district:
- Shooting victims grouped by `Fatal` / `Non-Fatal` since 2022
- Total shots fired incidents and ballistics-confirmed count since 2022

**Shootings dataset ID:** `73c7e069-701f-4910-986d-b950f46c91a1`
**Shots Fired dataset ID:** `c1e4e6ac-8a84-4b48-8a23-7b2645a32ede`
**Filter:** `district` (resolved via `NEIGHBORHOOD_TO_DISTRICT`)

**Key signals:**
- Total victims with Fatal / Non-Fatal breakdown → cited with exact numbers
- Fatal shootings → highest severity, flagged explicitly
- Shots fired total → signals active gun presence even without victims
- Data is district-level, not street-level → noted for buyer context

---

### Tool 8 — `fetch_green_space` (BPRD Trees + Open Space)

Runs three concurrent queries:
- Total tree count for the neighborhood via ArcGIS REST API
- Open space breakdown by type with acreage via CKAN
- Total usable recreational acres excluding cemeteries via CKAN

**BPRD Trees:** ArcGIS FeatureServer — `BPRD_Trees/FeatureServer/0/query`
**Filter:** `neighborhood` (resolved via `NEIGHBORHOOD_TO_BPRD`)

**Open Space dataset ID:** `61c0239f-c8fd-47de-8375-2405382ef37c`
**Filter:** `DISTRICT` (uses neighborhood name directly)

**Key signals:**
- Total tree count → canopy density proxy
- `Parks, Playgrounds & Athletic Fields` → most buyer-relevant open space type
- `Urban Wilds` → natural woodland, high value for nature-oriented buyers
- `Community Gardens` → engaged, sustainability-minded community
- `Cemeteries & Burying Grounds` → excluded from recreational acre total
- High tree count + high recreational acres → strong livability signal

---

## Summarization Node

After all eight fetch nodes complete, `summarize` is called with the full `context` list. It uses `llm_with_structure` (`ChatOpenAI` with `gpt-4o` and `.with_structured_output(NeighborhoodReport)`) to produce a validated nine-field response.

The system prompt instructs the LLM to act as a straight-talking buyer-focused analyst. It is explicitly told to:
- Flag red flags plainly without softening
- Cite specific numbers for every field
- Connect signals across datasets
- Address the buyer by household type and evaluate inventory for preferred property types when that context is provided
- Never carry data quality caveats into the `overall_verdict`
- Produce an `overall_verdict` of at least 150 words with a direct buyer-type recommendation

---

## Environment Variables Required

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required — used by `ChatOpenAI` to call GPT-4o |
| `LANGSMITH_API_KEY` | Optional — enables LangSmith tracing |
| `LANGSMITH_TRACING` | Optional — set to `true` to activate tracing |

---

## Valid Neighborhoods

The following neighborhood strings are supported. They must match exactly as entered:

```
Allston, Allston / Brighton, Back Bay, Beacon Hill, Brighton,
Charlestown, Dorchester, Downtown / Financial District, East Boston,
Fenway / Kenmore / Audubon Circle / Longwood, Greater Mattapan,
Hyde Park, Jamaica Plain, Mattapan, Mission Hill, Roslindale,
Roxbury, South Boston, South Boston / South Boston Waterfront,
South End, West Roxbury
```
