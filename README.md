# Neighborhood Analysis Backend

A FastAPI backend for a Neighborhood Analysis application. Users authenticate via Supabase, submit neighborhood search forms (neighborhood name, street, zip code), and retrieve or delete their saved searches. The backend is designed to eventually trigger a LangGraph AI agent that analyzes Boston neighborhood data using the Boston Open Data API.

**Live API:** [neighborhood-analysis-backend-production.up.railway.app](https://neighborhood-analysis-backend-production.up.railway.app)

## Tech Stack

| Layer            | Technology                                      |
| ---------------- | ----------------------------------------------- |
| Framework        | Python / FastAPI                                |
| Database & Auth  | Supabase (PostgreSQL + email/password auth)     |
| Token Validation | JWT via Supabase ECC (P-256) asymmetric keys    |
| Data Security    | Row Level Security (RLS) enforced at the DB     |
| Deployment       | Railway (auto-deploy on push to `main`)         |

## API Endpoints

All protected endpoints require an `Authorization: Bearer <jwt_token>` header. The JWT is issued by Supabase on login and verified server-side using `supabase.auth.get_user(token)` on every request.

| Method   | Path               | Auth       | Description                                      |
| -------- | ------------------ | ---------- | ------------------------------------------------ |
| `GET`    | `/health`          | Public     | Health check — returns `{"status": "ok"}`        |
| `POST`   | `/searches`        | Protected  | Save a neighborhood search for the current user  |
| `GET`    | `/searches`        | Protected  | Retrieve all saved searches for the current user |
| `DELETE` | `/searches/{id}`   | Protected  | Delete a saved search by ID (owner only)         |

**POST /searches request body:**

```json
{
  "neighborhood": "Back Bay",
  "street": "Newbury St",
  "zip_code": "02116"
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

Row Level Security ensures that even though all users' data lives in the same table, each user can only read, write, and delete their own rows. The `auth.uid()` function returns the UUID of the currently authenticated user from the JWT token. This is enforced at the **PostgreSQL level**, not just in application code.

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
```

### 4. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger UI.

## Supabase Database Setup

Run the following SQL blocks in the **Supabase SQL Editor** in order.

### Step 1 — Create the `saved_searches` table

```sql
CREATE TABLE saved_searches (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  neighborhood TEXT NOT NULL,
  street TEXT NOT NULL,
  zip_code TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### Step 2 — Enable RLS and add INSERT / SELECT policies

```sql
ALTER TABLE saved_searches ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert their own searches"
  ON saved_searches FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own searches"
  ON saved_searches FOR SELECT
  USING (auth.uid() = user_id);
```

### Step 3 — Add DELETE policy

```sql
CREATE POLICY "Users can delete their own searches"
  ON saved_searches FOR DELETE
  USING (auth.uid() = user_id);
```

## Deployment

- Deployed on **Railway** via GitHub integration.
- Auto-deploys on every push to the `main` branch.
- Environment variables (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) are set in Railway's service variables and are never committed to the repo.

## Roadmap

- Integrate a **LangGraph AI agent** triggered on neighborhood search submission
- Agent analyzes Boston Open Data (311 requests, crime data, property assessments)
- **Next.js or Vite frontend** with Supabase auth UI and search form
- Streaming agent responses to the frontend
- Chat interface for conversational follow-up on neighborhood reports
