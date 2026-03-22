# Agent Documentation

This folder contains two LangGraph-powered agents that together drive the AI features of The Hunt:

| Agent | File | Pattern | Trigger |
|---|---|---|---|
| **Neighborhood Analysis Agent** | `neighborhood_analysis.py` | Fan-out / fan-in (parallel) | `POST /searches` |
| **Chat Agent** | `chat_agent.py` | ReAct (tool-calling loop) | WebSocket `/ws/chat/{session_id}` |

Both agents use GPT-4o and query the Boston Open Data CKAN API. Neither uses a LangGraph checkpointer — the analysis agent is stateless by design, and the chat agent reconstructs conversation history from Supabase on every WebSocket message.

---

## Agent 1 — Neighborhood Analysis Agent

### Overview

A parallel fan-out/fan-in LangGraph graph. Eight fetch nodes run concurrently against the Boston Open Data API and the ArcGIS REST API, each writing their results to a shared `context` list. Once all eight complete, a single `summarize` node passes the full context to GPT-4o and returns a validated nine-field structured report. Supports optional buyer personalization via household type and property preferences.

### Architecture

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

All eight fetch nodes fan out from `START` simultaneously using LangGraph's parallel edge pattern. Each node appends its result to `context` via an `operator.add` reducer. LangGraph fans back in automatically once all eight nodes complete, then calls `summarize`.

### Graph State

```python
class State(TypedDict):
    neighborhood:         str          # e.g. "Jamaica Plain"
    street_name:          str          # e.g. "CREIGHTON ST" (uppercase, as in BPD dataset)
    zip_code:             str          # e.g. "02130"
    household_type:       str | None   # e.g. "partner", "family", "single" — optional
    property_preferences: list[str] | None  # e.g. ["Condo"] — max 2, optional
    context:   Annotated[list[str],  operator.add]  # populated by the eight fetch nodes
    raw_stats: Annotated[list[dict], operator.add]  # structured stats for UI display
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
    raw_stats:           list[dict]  # Structured crime, property, and 311 data for UI cards
```

`raw_stats` is populated by `fetch_crime`, `fetch_property`, and `fetch_311` at fetch time — no extra LLM calls required. The router injects `neighborhood_tiers` before saving to Supabase.

### Buyer Personalization

When `household_type` and/or `property_preferences` are provided, the `summarize` node injects them into the human message sent to GPT-4o:

```
Buyer profile: partner
Property preference: Condo (RESIDENTIAL CONDO)
```

Property preference labels are translated to their Boston assessment `LU_DESC` equivalents via `PROPERTY_PREF_TO_LU_DESC` so the LLM sees the exact strings present in the context block.

The system prompt instructs the LLM to:
- Address the buyer directly by household type in both `property_mix` and `overall_verdict`
- State explicitly whether strong or weak inventory exists for the preferred property types, citing actual counts
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

| Frontend Label | Boston Assessment LU_DESC |
|---|---|
| `Condo` | `RESIDENTIAL CONDO` |
| `Single Family` | `SINGLE FAM DWELLING` |
| `Two / Three Family` | `TWO-FAM DWELLING`, `THREE-FAM DWELLING` |
| `Small Apartment` | `APT 4-6 UNITS` |
| `Mid-Size Apartment` | `APT 7-30 UNITS` |
| `Mixed Use` | `RES /COMMERCIAL USE`, `COMM MULTI-USE` |

### Neighborhood Mappings

Two lookup dictionaries are defined at the top of `neighborhood_analysis.py`. Both use the same 21 canonical neighborhood keys.

**`NEIGHBORHOOD_TO_DISTRICT`** — Maps neighborhood names to BPD district codes. Used by `fetch_crime` and `fetch_gun_violence`.

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

**`NEIGHBORHOOD_TO_BPRD`** — Maps neighborhood names to BPRD Trees ArcGIS neighborhood values. Used by `fetch_green_space`.

| Agent Neighborhood | BPRD Value |
|---|---|
| Back Bay / Beacon Hill | `Back Bay/Beacon Hill` |
| Allston / Brighton / Brighton | `Allston-Brighton` |
| Fenway / Kenmore / Audubon Circle / Longwood | `Fenway/Longwood` |
| Downtown / Financial District | `Central Boston` |
| All others | Direct match |

### Data Sources

All CKAN sources are queried via:
```
https://data.boston.gov/api/3/action/datastore_search_sql?sql=<query>
```

The BPRD Trees source is queried via the ArcGIS REST API:
```
https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/BPRD_Trees/FeatureServer/0/query
```

All HTTP calls are made asynchronously using `httpx.AsyncClient` with a 15-second timeout.

### Fetch Nodes

**`fetch_311`** — Top 15 most frequent 311 complaint types for the neighborhood.
Dataset: `1a0b420d-99f1-4887-9851-990b2a5a6e17` · Filter: `neighborhood`
Also populates `raw_stats` with a filtered list (noise/admin types excluded) and total count.

**`fetch_crime`** — Top 15 most frequent crime types on the specific street in the current year.
Dataset: `b973d8cb-eeb2-4e7e-99da-c92938efc9c0` · Filter: `DISTRICT` + `STREET` + `YEAR = '2026'`
Also populates `raw_stats` with the top 6 meaningful offenses (procedural types excluded).

**`fetch_property`** — Top 15 most frequent property use types in the zip code.
Dataset: `ee73430d-96c0-423e-ad21-c4cfb54c8961` · Filter: `ZIP_CODE`
Also populates `raw_stats` with counts bucketed by the six frontend property preference categories.

**`fetch_permits`** — Building permit activity by worktype, ordered by total declared investment value, rolling 2-year window.
Dataset: `6ddcd912-32a0-43df-9908-63574f8c7e77` · Filter: `zip` + `issued_date >= CURRENT_DATE - INTERVAL '2 years'`

**`fetch_entertainment`** — Entertainment unit type breakdown for the zip code (venue count + total units).
Dataset: `1c4c1f7c-9a2a-4f4f-85a7-d3462c6bc9cb` · Filter: `zip` + `status = 'Active'`

**`fetch_traffic_safety`** — Runs three concurrent queries for the specific street: crash counts by mode type since 2022, top 5 hotspot intersections since 2022, all-time fatality records.
Crash dataset: `e4bfe397-6bfc-49c5-9367-c879fac7401d` · Fatality dataset: `92f18923-d4ec-4c17-9405-4e0da63e1d6c`
Filter: `street OR xstreet1 OR xstreet2 = street_name`

**`fetch_gun_violence`** — Runs two concurrent queries for the BPD district: shooting victims (Fatal/Non-Fatal) since 2022, shots fired total and ballistics-confirmed count since 2022.
Shootings dataset: `73c7e069-701f-4910-986d-b950f46c91a1` · Shots fired dataset: `c1e4e6ac-8a84-4b48-8a23-7b2645a32ede`
Filter: `district` (resolved via `NEIGHBORHOOD_TO_DISTRICT`)

**`fetch_green_space`** — Runs three concurrent queries: total tree count via ArcGIS, open space breakdown by type with acreage, total usable recreational acres excluding cemeteries.
BPRD Trees: ArcGIS FeatureServer · Open space dataset: `61c0239f-c8fd-47de-8375-2405382ef37c`

### Summarization Node

After all eight fetch nodes complete, `summarize` calls `llm_with_structure` (`ChatOpenAI` with `gpt-4o` and `.with_structured_output(NeighborhoodReport)`) with the full `context` list. The system prompt instructs GPT-4o to act as a straight-talking buyer-focused analyst — flagging red flags plainly, citing specific numbers, blending live data with training-data knowledge of Boston's neighborhoods, and producing an `overall_verdict` of at least 250 words with a direct buyer-type recommendation.

---

## Agent 2 — Chat Agent

### Overview

A streaming ReAct agent built on LangGraph's `MessagesState`. The assistant node decides which tools to call based on the user's question — it only fetches the data that is relevant to the query rather than running all tools on every turn. The agent is stateless: no LangGraph checkpointer is attached. Conversation history is persisted in Supabase (`chat_messages` table) and reconstructed by the WebSocket handler on every incoming message.

### Architecture

```
START → assistant ──► tools_condition ──► ToolNode ──► assistant ──► ... ──► END
                  └──► END (if no tool call)
```

The ReAct loop continues until the assistant produces a response with no tool calls, at which point `tools_condition` routes to `END`.

### Tools (7)

| Tool | Input | Data Source | Use When |
|---|---|---|---|
| `fetch_311` | `neighborhood` | Boston 311 dataset | Complaints, noise, trash, code enforcement, quality of life |
| `fetch_crime` | `neighborhood`, `street` | BPD crime dataset | Safety or crime on a specific street |
| `fetch_property` | `zip_code` | Boston property assessment | Housing types, property mix, condos, rental stock |
| `fetch_permits` | `zip_code` | Building permits dataset | Development, construction, renovation, investment trends |
| `fetch_entertainment` | `zip_code` | Entertainment licenses dataset | Nightlife, bars, restaurants, noise levels |
| `fetch_traffic_safety` | `street` | Vision Zero crash + fatality datasets | Traffic crashes, pedestrian/cyclist safety, dangerous intersections |
| `fetch_gun_violence` | `neighborhood` | BPD shootings + shots fired datasets | Shootings, gun violence, district-level safety |

All tools are `@tool`-decorated async functions. The LLM receives their docstrings as tool descriptions and decides at runtime which to call based on the user's question. Multiple tools can be called in a single turn.

### State

Uses LangGraph's built-in `MessagesState` — a list of LangChain message objects (`HumanMessage`, `AIMessage`, `ToolMessage`). The WebSocket handler reconstructs this from Supabase on every incoming message and passes it directly to `stream_chat`.

### Zip Code Resolution

Many tools require a zip code. Rather than asking the user, the system prompt embeds a full `neighborhood → zip code` mapping. Rules:
- Single-zip neighborhoods (e.g. Back Bay → `02116`): resolved automatically, tool called immediately.
- Multi-zip neighborhoods (e.g. Dorchester has 4 zips): agent asks the user which part they mean before calling the tool.

### Streaming

Tokens are streamed via `graph.astream_events(version="v2")`. Only `on_chat_model_stream` events from the `assistant` node are yielded — tool call internals are filtered out. The WebSocket handler in `routers/ws.py` pipes these tokens to the client one at a time and sends `[DONE]` when the turn is complete.

```python
async def stream_chat(messages: list, config: dict):
    async for event in graph.astream_events({"messages": messages}, config, version="v2"):
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node") == "assistant"
        ):
            token = event["data"]["chunk"].content
            if token:
                yield token
```

### History Reconstruction

The chat agent has no memory between turns. On every WebSocket message the handler:
1. Saves the incoming human message to Supabase
2. Fetches the full session history from `chat_messages` ordered by `created_at`
3. Reconstructs it as a list of `HumanMessage` / `AIMessage` objects
4. Passes the full list to `stream_chat`

This means the agent always has complete context without a checkpointer.

---

## Shared Environment Variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required — used by `ChatOpenAI` to call GPT-4o in both agents |
| `LANGSMITH_API_KEY` | Optional — enables LangSmith tracing |
| `LANGSMITH_TRACING` | Optional — set to `true` to activate tracing |

---

## Valid Neighborhoods

The following 21 neighborhood strings are supported by both agents. They must match exactly:

```
Allston, Allston / Brighton, Back Bay, Beacon Hill, Brighton,
Charlestown, Dorchester, Downtown / Financial District, East Boston,
Fenway / Kenmore / Audubon Circle / Longwood, Greater Mattapan,
Hyde Park, Jamaica Plain, Mattapan, Mission Hill, Roslindale,
Roxbury, South Boston, South Boston / South Boston Waterfront,
South End, West Roxbury
```
