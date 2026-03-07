# Neighborhood Analysis Agent

A LangGraph-powered agent that analyzes Boston neighborhoods by making three parallel calls to the Boston Open Data API, then synthesizing the results into a structured report using GPT-4o.

---

## Architecture

```
START
  ├── fetch_311       ─┐
  ├── fetch_crime     ─┼── (parallel) ──► summarize ──► END
  └── fetch_property  ─┘
```

The three fetch nodes run in parallel using LangGraph's fan-out/fan-in pattern. Each node writes its result to a shared `context` list in state using an `operator.add` reducer, which appends all three results without conflict. Once all three finish, LangGraph automatically fans back in to the `summarize` node, which passes the full context to GPT-4o and returns a structured four-field report.

---

## Graph State

```python
class State(TypedDict):
    neighborhood: str       # e.g. "Back Bay"
    street_name:  str       # e.g. "BOYLSTON ST"  (uppercase, as it appears in BPD dataset)
    zip_code:     str       # e.g. "02116"
    context:      Annotated[list[str], operator.add]  # populated by the three fetch nodes
```

The `context` key uses `operator.add` as a reducer so all three parallel nodes can safely append to the same list in the same graph step.

**Output:**

```python
class OutputState(TypedDict):
    requests_311:    str   # Analysis of 311 service request patterns
    crime_safety:    str   # Analysis of crime incidents on the street
    property_mix:    str   # Analysis of property types in the zip code
    overall_verdict: str   # Synthesized buyer-focused verdict
```

---

## Data Sources

All three data sources are from the [Boston Open Data Portal](https://data.boston.gov) and are queried via the CKAN `datastore_search_sql` API endpoint:

```
https://data.boston.gov/api/3/action/datastore_search_sql?sql=<query>
```

---

### Tool 1 — `fetch_311` (311 Service Requests)

Fetches the top 15 most frequent 311 complaint types filed in the neighborhood.

**Dataset ID:** `1a0b420d-99f1-4887-9851-990b2a5a6e17`

**Input:** `neighborhood` (e.g. `Back Bay`)

**Example query:**
```sql
SELECT "type", COUNT(*) as count
FROM "1a0b420d-99f1-4887-9851-990b2a5a6e17"
WHERE "neighborhood" = 'Back Bay'
GROUP BY "type"
ORDER BY count DESC
LIMIT 15
```

**Browser URL:**
```
https://data.boston.gov/api/3/action/datastore_search_sql?sql=SELECT "type", COUNT(*) as count FROM "1a0b420d-99f1-4887-9851-990b2a5a6e17" WHERE "neighborhood" = 'Back Bay' GROUP BY "type" ORDER BY count DESC LIMIT 15
```

**Example result (Back Bay):**

| type | count |
|---|---|
| Request for Snow Plowing | 339 |
| Parking Enforcement | 257 |
| Unshoveled Sidewalk | 189 |
| Improper Storage of Trash (Barrels) | 174 |
| CE Collection | 114 |
| Request for Pothole Repair | 103 |
| Requests for Street Cleaning | 76 |
| Traffic Signal Inspection | 56 |
| Needle Pickup | 56 |
| Pick up Dead Animal | 47 |
| Sidewalk Repair (Make Safe) | 42 |
| Missed Trash/Recycling/Yard Waste/Bulk Item | 38 |
| Poor Conditions of Property | 38 |
| Sign Repair | 24 |
| Encampments | 21 |

**Notable signal types the LLM is instructed to flag:**
- `CE Collection` → Code Enforcement Collection (city pursuing code violations)
- `Needle Pickup` / `Encampments` → visible substance abuse or homelessness
- `Unsatisfactory Living Conditions` / `Heat - Excessive/Insufficient` → housing quality issues
- `Space Savers` → parking scarcity (post-snowstorm parking disputes)
- `Illegal Dumping` / `Abandoned Vehicles` → neighborhood neglect

---

### Tool 2 — `fetch_crime` (Crime & Safety)

Fetches the top 15 most frequent crime/incident types reported on the specific street in the current year.

**Dataset ID:** `b973d8cb-eeb2-4e7e-99da-c92938efc9c0`

**Input:** `street_name` + BPD district code (resolved from `neighborhood` via `NEIGHBORHOOD_TO_DISTRICT` mapping)

**Neighborhood → BPD District mapping:**

| Neighborhood | District |
|---|---|
| Back Bay | D4 |
| South End | D4 |
| Fenway / Kenmore / Audubon Circle / Longwood | D4 |
| Beacon Hill | A1 |
| Downtown / Financial District | A1 |
| Charlestown | A15 |
| East Boston | A7 |
| Roxbury | B2 |
| Mattapan / Greater Mattapan | B3 |
| Dorchester | C11 |
| South Boston / South Boston Waterfront | C6 |
| Jamaica Plain / Mission Hill / Roslindale | E13 |
| West Roxbury | E5 |
| Hyde Park | E18 |
| Allston / Brighton | D14 |

**Example query:**
```sql
SELECT "OFFENSE_DESCRIPTION", COUNT(*) as count
FROM "b973d8cb-eeb2-4e7e-99da-c92938efc9c0"
WHERE "DISTRICT" = 'D4'
AND "STREET" = 'BOYLSTON ST'
AND "YEAR" = '2026'
GROUP BY "OFFENSE_DESCRIPTION"
ORDER BY count DESC
LIMIT 15
```

**Browser URL:**
```
https://data.boston.gov/api/3/action/datastore_search_sql?sql=SELECT "OFFENSE_DESCRIPTION", COUNT(*) as count FROM "b973d8cb-eeb2-4e7e-99da-c92938efc9c0" WHERE "DISTRICT" = 'D4' AND "STREET" = 'BOYLSTON ST' AND "YEAR" = '2026' GROUP BY "OFFENSE_DESCRIPTION" ORDER BY count DESC LIMIT 15
```

**Example result (Boylston St, District D4, 2026):**

| OFFENSE_DESCRIPTION | count |
|---|---|
| LARCENY SHOPLIFTING | 123 |
| SICK ASSIST | 22 |
| INVESTIGATE PERSON | 18 |
| ASSAULT - SIMPLE | 11 |
| INVESTIGATE PROPERTY | 8 |
| LARCENY THEFT FROM BUILDING | 7 |
| TOWED MOTOR VEHICLE | 5 |
| VANDALISM | 5 |
| TRESPASSING | 4 |
| BURGLARY - COMMERICAL | 4 |
| ROBBERY | 3 |
| PROPERTY - LOST/ MISSING | 3 |
| M/V - LEAVING SCENE - PROPERTY DAMAGE | 3 |
| SICK ASSIST - DRUG RELATED ILLNESS | 2 |
| LICENSE PREMISE VIOLATION | 2 |

**Notable signal types the LLM is instructed to flag:**
- `INVESTIGATE PERSON` / `INVESTIGATE PROPERTY` → procedural, not actual crimes
- Auto theft, drug offenses, robbery, threats → flagged explicitly
- Low counts on a residential street → noted as reassuring

> **Note:** `YEAR` is hardcoded to `'2026'` in the query. If the BPD dataset hasn't been updated with current year data yet, the node will return a "no records found" message and the LLM will note the data gap.

---

### Tool 3 — `fetch_property` (Property Assessment)

Fetches the top 15 most frequent property use types in the zip code from the Boston property assessment dataset.

**Dataset ID:** `ee73430d-96c0-423e-ad21-c4cfb54c8961`

**Input:** `zip_code` (e.g. `02116`)

**Example query:**
```sql
SELECT "LU_DESC", COUNT(*) as count
FROM "ee73430d-96c0-423e-ad21-c4cfb54c8961"
WHERE "ZIP_CODE" = '02116'
GROUP BY "LU_DESC"
ORDER BY count DESC
LIMIT 15
```

**Browser URL:**
```
https://data.boston.gov/api/3/action/datastore_search_sql?sql=SELECT "LU_DESC", COUNT(*) as count FROM "ee73430d-96c0-423e-ad21-c4cfb54c8961" WHERE "ZIP_CODE" = '02116' GROUP BY "LU_DESC" ORDER BY count DESC LIMIT 15
```

**Example result (02116 — Back Bay):**

| LU_DESC | count |
|---|---|
| RESIDENTIAL CONDO | 6832 |
| CONDO MAIN | 839 |
| CONDO PARKING (RES) | 479 |
| SINGLE FAM DWELLING | 318 |
| APT 7-30 UNITS | 148 |
| APT 4-6 UNITS | 113 |
| RET/WHSL/SERVICE | 107 |
| COMM MULTI-USE | 91 |
| TWO-FAM DWELLING | 83 |
| THREE-FAM DWELLING | 79 |
| RES /COMMERCIAL USE | 77 |
| OFFICE CONDO | 60 |
| OTHER EXEMPT BLDG | 49 |
| RETAIL CONDO | 42 |
| OFFICE 3-9 STORY | 31 |

**Notable property types the LLM is instructed to flag:**
- `CONDO PARKING (RES)` → parking sold separately, extra cost for buyers
- `SUBSD HOUSING S-8` → Section 8 subsidized housing
- `CITY OF BOSTON` / `BOST REDEVELOP AUTH` / `BOS HOUSING AUTHOR` → publicly owned land
- `RES LAND (Unusable)` → vacant lots, possible blight or future development
- Two- and three-family home dominance → rental-heavy neighborhood

---

## Summarization Node

After all three fetch nodes complete, `summarize` is called with the full `context` list. It uses `llm_with_structure` (`ChatOpenAI` with `gpt-4o` and `.with_structured_output(NeighborhoodReport)`) to produce a validated four-field response.

The system prompt instructs the LLM to act as a straight-talking buyer-focused analyst — not a reassuring generalist. It is explicitly told to flag red flags, connect dots across the three datasets, and be specific rather than vague.

The structured output schema enforces that the response always contains exactly:
- `requests_311`
- `crime_safety`
- `property_mix`
- `overall_verdict`

---

## Environment Variables Required

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required — used by `ChatOpenAI` to call GPT-4o |
| `LANGSMITH_API_KEY` | Optional — enables LangSmith tracing |
| `LANGSMITH_TRACING` | Optional — set to `true` to activate tracing |

---

## Valid Neighborhoods

The `fetch_crime` node resolves the BPD district from the neighborhood name. Only the following neighborhood strings are supported (must match exactly as they appear in the 311 dataset):

```
Allston, Allston / Brighton, Back Bay, Beacon Hill, Brighton,
Charlestown, Dorchester, Downtown / Financial District, East Boston,
Fenway / Kenmore / Audubon Circle / Longwood, Greater Mattapan,
Hyde Park, Jamaica Plain, Mattapan, Mission Hill, Roslindale,
Roxbury, South Boston, South Boston / South Boston Waterfront,
South End, West Roxbury
```

If the neighborhood string does not match any key in `NEIGHBORHOOD_TO_DISTRICT`, `fetch_crime` returns a "could not resolve district" message and the crime section of the report will note the data gap.
