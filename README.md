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

**POST /searches request body:**

```json
{
  "neighborhood": "Back Bay",
  "street": "Newbury St",
  "zip_code": "02116",
  "household_type": "partner",
  "property_preferences": ["Condo"]
}
```

## Project Structure

```
neighborhood-analysis-backend/
├── main.py               # FastAPI app, CORS middleware, router registration
├── auth.py               # JWT verification dependency using Supabase get_user()
├── database.py           # Supabase client initialization
├── models.py             # Pydantic v2 request/response models
├── routers/
│   └── searches.py       # Endpoint handlers, per-request Supabase client
├── agent/
│   ├── neighborhood_analysis.py  # Parallel 8-node LangGraph analysis agent
│   └── chat_agent.py             # Streaming conversational chat agent
├── requirements.txt
├── Procfile              # Railway start command
└── .env                  # Local environment variables (gitignored)
```

## Authentication

- Supabase issues JWT tokens signed with **ECC (P-256) asymmetric keys**.
- The backend verifies tokens by calling `supabase.auth.get_user(token)` on every protected request — this validates the token server-side.
- The backend **never trusts `user_id` from the request body** — it always derives it from the verified JWT token.
- A fresh Supabase client is created per request with the user's token to avoid global state mutation in concurrent environments.

## Row Level Security

Row Level Security ensures that even though all users' data lives in the same tables, each user can only read, write, and delete their own rows. The `auth.uid()` function returns the UUID of the currently authenticated user from the JWT token. This is enforced at the **PostgreSQL level**, not just in application code. All three tables — `saved_searches`, `chat_sessions`, and `chat_messages` — have RLS enabled with identical per-user policies.

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

The app uses three tables. Run all SQL blocks below in the **Supabase SQL Editor** in order.

---

### Table 1 — `saved_searches`

Stores neighborhood search submissions and the AI-generated analysis report for each one.

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

Each row represents one conversation thread. The `title` is set to the user's first message (truncated). Users can have multiple sessions and revisit old ones.

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

## Deployment

- Deployed on **Railway** via GitHub integration.
- Auto-deploys on every push to the `main` branch.
- Environment variables (`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `OPENAI_API_KEY`) are set in Railway's service variables and are never committed to the repo.

---

## Roadmap

### UI
- **Update Login / Sign-Up UI** with this component: https://21st.dev/community/components/easemize/sign-in/default

### Agent & State
- **Enrich state with parallel and intersecting streets** — explore whether the state passed to the agent can be expanded so that alongside the user-submitted street name, streets that are parallel and intersecting with it are automatically added. This would allow the agent to build a more complete picture of the immediate area rather than a single corridor.
- **Figure out if an API exists to turn street names into coordinates** — a geocoding API (e.g. Mapbox Geocoding, Google Maps Geocoding, or Boston's own SAM address dataset) could convert a street name + neighborhood into lat/long bounds. This is a prerequisite for the map visualization work and for the expanded street state.
- **Adding more streets will allow a more complete crime picture** — once parallel and intersecting streets are included in the state, the crime and traffic safety tools can query across all of them. The resulting data could either align with or contrast against the broader neighborhood-level picture, giving the buyer a much richer signal.

### Data & Database
- **Add a Supabase table for neighborhood-level georeferenced records** — create a table for dumping 311 request and crime records that include coordinates (lat/long). This table would be populated at analysis time and queried by the map visualization layer rather than storing large coordinate arrays in the `analysis` JSONB field of `saved_searches`.

### Scoring
- **Add a rating to each section** — each of the 8 analysis sections (311, crime, property mix, permits, entertainment, traffic safety, gun violence, green space) could be assigned a score indicating how the neighborhood stacks up for that category relative to Boston as a whole. This could be achieved programmatically using the raw data counts and thresholds, without requiring additional LLM calls.

### Prompting
- **Further prompt improvements** — more signal can be extracted from the data the LLMs are trained on. GPT-4o has knowledge of Boston's neighborhoods up to its training cutoff and can be prompted more effectively to blend live data with that background knowledge to produce richer, more contextual analysis across all eight sections.
