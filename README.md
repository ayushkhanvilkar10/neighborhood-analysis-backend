# Neighborhood Analysis Backend

A FastAPI backend for a Neighborhood Analysis application. Users authenticate via Supabase, submit neighborhood search forms, and receive AI-generated neighborhood reports powered by a LangGraph agent. The backend also supports a streaming chat interface where users can have multi-turn conversations with an AI assistant about Boston neighborhoods.

**Live API:** [neighborhood-analysis-backend-production.up.railway.app](https://neighborhood-analysis-backend-production.up.railway.app)

## Tech Stack

| Layer            | Technology                                      |
| ---------------- | ----------------------------------------------- |
| Framework        | Python / FastAPI                                |
| Database & Auth  | Supabase (PostgreSQL + email/password auth)     |
| Token Validation | JWT via Supabase ECC (P-256) asymmetric keys    |
| Data Security    | Row Level Security (RLS) enforced at the DB     |
| AI               | LangGraph + GPT-4o                              |
| Deployment       | Railway (auto-deploy on push to `main`)         |

## API Endpoints

All protected endpoints require an `Authorization: Bearer <jwt_token>` header. The JWT is issued by Supabase on login and verified server-side using `supabase.auth.get_user(token)` on every request.

| Method   | Path               | Auth       | Description                                      |
| -------- | ------------------ | ---------- | ------------------------------------------------ |
| `GET`    | `/health`          | Public     | Health check — returns `{"status": "ok"}`        |
| `POST`   | `/searches`        | Protected  | Run agent analysis and save search               |
| `GET`    | `/searches`        | Protected  | Retrieve all saved searches for the current user |
| `DELETE` | `/searches/{id}`   | Protected  | Delete a saved search by ID (owner only)         |
| `GET`    | `/preferences`     | Protected  | Retrieve the current user's saved preferences    |
| `PUT`    | `/preferences`     | Protected  | Create or update user preferences (upsert)       |
| `DELETE` | `/preferences`     | Protected  | Reset user preferences (delete row)              |

**POST /searches request body:**

```json
{
  "neighborhood": "Back Bay",
  "street": "NEWBURY ST",
  "zip_code": "02116",
  "household_type": "Couple / Partner",
  "property_preferences": ["Condo"]
}
```

**POST /searches response — `analysis` object:**

The `analysis` field in the response (and in saved searches returned by `GET /searches`) contains:

```json
{
  "requests_311":        "string — LLM analysis of 311 complaint patterns",
  "crime_safety":        "string — LLM analysis of crime incidents on the street",
  "property_mix":        "string — LLM analysis of property types in the zip code",
  "permit_activity":     "string — LLM analysis of building permit investment",
  "entertainment_scene": "string — LLM analysis of entertainment license types",
  "traffic_safety":      "string — LLM analysis of crashes and fatalities",
  "gun_violence":        "string — LLM analysis of shootings and shots fired",
  "green_space":         "string — LLM analysis of trees and open space",
  "overall_verdict":     "string — synthesized buyer-focused verdict (250+ words)",
  "raw_stats": [
    { "section": "crime_safety",  "data": [{ "offense": "...", "count": 0 }] },
    { "section": "property_mix",  "data": { "Condo": 0 }, "total": 0 },
    { "section": "requests_311",  "data": [{ "type": "...", "count": 0 }], "total": 0 }
  ],
  "neighborhood_tiers": {
    "crime":          "High | Moderate | Low",
    "complaints_311": "High | Moderate | Low"
  }
}
```

`raw_stats` is populated by the parallel fetch nodes at analysis time — no extra LLM calls required. `neighborhood_tiers` is a static lookup from `agent/neighborhood_tiers.json` injected at the router level before saving to Supabase.

**PUT /preferences request body:**

Both fields are optional. The frontend auto-saves preferences on change so they persist across sessions.

```json
{
  "household_type": "Couple / Partner",
  "property_preferences": ["Condo", "Single Family"]
}
```

Allowed `household_type` values: `"Living solo"`, `"Couple / Partner"`, `"Family with kids"`, `"Retiree / Empty nester"`, `"Investor"`.

Allowed `property_preferences` values (max 2): `"Condo"`, `"Single Family"`, `"Two / Three Family"`, `"Small Apartment"`, `"Mid-Size Apartment"`, `"Mixed Use"`.

**GET /preferences response:**

```json
{
  "household_type": "Couple / Partner",
  "property_preferences": ["Condo", "Single Family"],
  "updated_at": "2026-03-27T00:20:17.052549Z"
}
```

Returns empty defaults (`null` fields, no `updated_at`) if no preferences have been saved yet.

## Project Structure

```
neighborhood-analysis-backend/
├── main.py                          # FastAPI app, CORS middleware, router registration
├── auth.py                          # JWT verification dependency using Supabase get_user()
├── database.py                      # Supabase client initialization
├── models.py                        # Pydantic v2 request/response models
├── routers/
│   ├── searches.py                  # Endpoint handlers — loads neighborhood_tiers.json at startup
│   └── preferences.py               # User preferences CRUD (GET, PUT, DELETE)
├── agent/
│   ├── neighborhood_analysis.py     # Parallel 8-node LangGraph analysis agent
│   ├── chat_agent.py                # Streaming ReAct chat agent (7 tools)
│   └── neighborhood_tiers.json      # Static crime + 311 tier lookup for all 21 neighborhoods
├── Neighborhood_Rankings.md         # Three neighborhood rankings with methodology and SQL queries
├── requirements.txt
├── Procfile                         # Railway start command
└── .env                             # Local environment variables (gitignored)
```

## Neighborhood Tiers

`agent/neighborhood_tiers.json` maps all 21 canonical neighborhood values to pre-computed tier scores across two dimensions:

- **`crime`** — based on knowledge-adjusted BPD crime data (Aggravated Assault, Threats, Robbery, Drugs, Residential Burglary)
- **`complaints_311`** — based on 2026 YTD Boston 311 data filtered to high-severity complaint types (CE Collection, Needle Pickup, Encampments, Heat complaints, Unsatisfactory Living Conditions)

Tiers: `"High"`, `"Moderate"`, `"Low"`. These are loaded once at server startup in `routers/searches.py` and injected into `analysis_dict` before saving to Supabase, making them available on both `POST /searches` (new analysis) and `GET /searches` (saved searches).

The full ranking methodology, SQL queries used, and tier tables are documented in `Neighborhood_Rankings.md`.

## Authentication

- Supabase issues JWT tokens signed with **ECC (P-256) asymmetric keys**.
- The backend verifies tokens by calling `supabase.auth.get_user(token)` on every protected request.
- The backend **never trusts `user_id` from the request body** — it always derives it from the verified JWT token.
- A fresh Supabase client is created per request with the user's token to avoid global state mutation in concurrent environments.

## Row Level Security

Row Level Security ensures that even though all users' data lives in the same tables, each user can only read, write, and delete their own rows. The `auth.uid()` function returns the UUID of the currently authenticated user from the JWT token. This is enforced at the **PostgreSQL level**, not just in application code. All four tables — `saved_searches`, `chat_sessions`, `chat_messages`, and `user_preferences` — have RLS enabled with per-user policies. The `user_preferences` table additionally has an UPDATE policy since preferences are edited over time, unlike searches and messages which are write-once.

## Local Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd neighborhood-analysis-backend
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_api_key
```

### 4. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger UI.

## Supabase Database Setup

The app uses four tables. Run all SQL blocks below in the **Supabase SQL Editor** in order.

---

### Table 1 — `saved_searches`

Stores neighborhood search submissions and the AI-generated analysis report for each one. The `analysis` JSONB column stores the full 9-field LLM report, `raw_stats`, and `neighborhood_tiers`.

```sql
CREATE TABLE saved_searches (
  id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id      UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  neighborhood TEXT NOT NULL,
  street       TEXT NOT NULL,
  zip_code     TEXT NOT NULL,
  analysis     JSONB,
  created_at   TIMESTAMPTZ DEFAULT now()
);
```

```sql
ALTER TABLE saved_searches ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert their own searches"
  ON saved_searches FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own searches"
  ON saved_searches FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own searches"
  ON saved_searches FOR DELETE
  USING (auth.uid() = user_id);
```

---

### Table 2 — `chat_sessions`

Each row represents one conversation thread. The `title` is set to the user's first message (truncated). Users can have multiple sessions and revisit old ones from the sidebar navigation.

```sql
CREATE TABLE chat_sessions (
  id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id    UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  title      TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

```sql
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert their own sessions"
  ON chat_sessions FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own sessions"
  ON chat_sessions FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own sessions"
  ON chat_sessions FOR DELETE
  USING (auth.uid() = user_id);
```

---

### Table 3 — `chat_messages`

Stores individual messages within a chat session. One row per message. The `role` field distinguishes user input from AI responses. Deleting a session automatically cascades and removes all its messages.

```sql
CREATE TABLE chat_messages (
  id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE NOT NULL,
  user_id    UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  role       TEXT NOT NULL CHECK (role IN ('human', 'ai')),
  content    TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

```sql
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert their own messages"
  ON chat_messages FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own messages"
  ON chat_messages FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own messages"
  ON chat_messages FOR DELETE
  USING (auth.uid() = user_id);
```

---

### Table 4 — `user_preferences`

Stores per-user household type and property preferences so they don't need to be re-entered on every search. One row per user, enforced by a `UNIQUE` constraint on `user_id`. The backend uses an upsert pattern (insert on first save, update thereafter). The `household_type` column has a `CHECK` constraint limiting it to the five allowed values. The `property_preferences` column is a Postgres text array with a max length of 2. Unlike the other tables, this table has an `updated_at` timestamp instead of `created_at` since preferences are edited over time, and includes an UPDATE RLS policy.

```sql
CREATE TABLE user_preferences (
  id                    UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id               UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
  household_type        TEXT CHECK (household_type IN (
                          'Living solo',
                          'Couple / Partner',
                          'Family with kids',
                          'Retiree / Empty nester',
                          'Investor'
                        )),
  property_preferences  TEXT[] DEFAULT '{}' CHECK (array_length(property_preferences, 1) <= 2),
  updated_at            TIMESTAMPTZ DEFAULT now()
);
```

```sql
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert their own preferences"
  ON user_preferences FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own preferences"
  ON user_preferences FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own preferences"
  ON user_preferences FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own preferences"
  ON user_preferences FOR DELETE
  USING (auth.uid() = user_id);
```

---

## Deployment

- Deployed on **Railway** via GitHub integration.
- Auto-deploys on every push to the `main` branch.
- Environment variables (`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `OPENAI_API_KEY`) are set in Railway's service variables and are never committed to the repo.

---

## Roadmap

### UI
- ~~**Add login page UI component**~~ ✅ — Done. Sign-in/sign-up form styled with the Forest Anchor palette: white card, teal (`#649E97`) borders, `#F8FBFA` input backgrounds, forest green (`#006B4E`) focus states and CTA button.
- **Simplify the neighborhood analyzer search form** — once user preferences are persisted in the database (see Data & Database below), the search form should only require neighborhood, street, and zip code. Household type and property preferences should be read from the user's saved profile automatically.
- ~~**Finalize app color palette**~~ ✅ — Done. Adopted the "Forest Anchor" palette across all frontend components: forest green sidebar (`#006B4E`), white cards, teal (`#649E97`) border system, forest green CTA buttons, green progress bar ramp. Applied to: sidebar (`app-nav.tsx`), dashboard form, analysis cards, stat tiles, property mix card, 311 card, chat page (light user bubbles, glass AI bubbles, light input bar), and sign-in page.
- **Change the logo for the app** — design or source a new logo that better represents the product.
- **Property Preferences should live as a separate modal** — move property preferences out of the current form and into its own dedicated modal component.
- **Fix errors related to conversation history tab** — debug and resolve issues in the chat history / conversation sidebar.
- **Update map with actual location markers** — add pins / markers to the map for crime incidents and gun shooting locations.

### Agent & State
- **Integrate a Blue Bikes tool** — add a ninth parallel fetch node that queries the Blue Bikes dataset for station locations and dock availability in the neighborhood. This adds a transit and mobility signal to the analysis, relevant to buyers who commute by bike or value car-free infrastructure.
- **Add Food Licenses API** — either as a standalone parallel fetch node or incorporated into the existing entertainment licenses tool to surface restaurant and food establishment data for the neighborhood.
- **Enrich state with parallel and intersecting streets** — explore whether the state passed to the agent can be expanded so that alongside the user-submitted street name, streets that are parallel and intersecting with it are automatically added. This would allow the agent to build a more complete picture of the immediate area rather than a single corridor.
- **Figure out if an API exists to turn street names into coordinates** — a geocoding API (e.g. Mapbox Geocoding, Google Maps Geocoding, or Boston's own SAM address dataset) could convert a street name + neighborhood into lat/long bounds. This is a prerequisite for the map visualization work and for the expanded street state.
- **Adding more streets will allow a more complete crime picture** — once parallel and intersecting streets are included in the state, the crime and traffic safety tools can query across all of them, giving the buyer a much richer signal.
- **Create use cases for ReAct Chat agent that involve use of multiple tools** — design and document multi-tool scenarios (e.g. comparing crime + green space across neighborhoods) to stress-test and showcase the chat agent's reasoning capabilities.

### Data & Database
- ~~**Add a `user_preferences` Supabase table**~~ ✅ — Done. The `user_preferences` table persists household type and property preferences per user, keyed by `user_id` with RLS enforced. The frontend auto-saves preferences on change via `PUT /preferences` and loads them on dashboard mount via `GET /preferences`.
- **Add a Supabase table for neighborhood-level georeferenced records** — create a table for dumping 311 request and crime records that include coordinates (lat/long). This table would be populated at analysis time and queried by the map visualization layer rather than storing large coordinate arrays in the `analysis` JSONB field of `saved_searches`.
- **Automate neighborhood tier computation via an agentic pipeline** — the current `neighborhood_tiers.json` is a static file. Long-term, a LangGraph pipeline on a Railway cron job (weekly) should query the Boston Open Data endpoints, apply the consolidation and correction logic documented in `Neighborhood_Rankings.md`, and upsert results into a `neighborhood_tiers` Supabase table. The LLM correction node handles the knowledge-based reranking (e.g. district boundary distortions, label inconsistencies) that cannot be derived mechanically from raw counts alone.

### Scoring
- **Add a rating to each section** — each of the 8 analysis sections (311, crime, property mix, permits, entertainment, traffic safety, gun violence, green space) could be assigned a score indicating how the neighborhood stacks up for that category relative to Boston as a whole. The `neighborhood_tiers.json` data is the foundation for this — the tier values can be surfaced as visual badges in the analysis report.

### Prompting
- **Inject persisted user preferences into the agent prompt** — at analysis time, fetch the user's saved household type and property preferences from the `user_preferences` table and pass them to the summarization node. This removes the need for the frontend form to collect them on every search and ensures the agent always has buyer context.
- **Further prompt improvements** — more signal can be extracted from the data the LLMs are trained on. GPT-4o has knowledge of Boston's neighborhoods up to its training cutoff and can be prompted more effectively to blend live data with that background knowledge to produce richer, more contextual analysis across all eight sections.
- **Make section content more accessible** — the current analysis sections quote jargon and raw numbers directly from API responses. Rewrite prompts so the LLM translates data into plain language that an average user (not a data analyst) can easily understand.
- **Make chat agent responses more descriptive** — the ReAct chat agent currently leans too heavily on quoting raw API data. Improve prompts so it synthesizes and explains findings in a conversational, human-friendly way rather than parroting numbers.
