import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL:
    raise EnvironmentError(
        "SUPABASE_URL is not set. "
        "Please add it to your .env file or environment variables."
    )

if not SUPABASE_ANON_KEY:
    raise EnvironmentError(
        "SUPABASE_ANON_KEY is not set. "
        "Please add it to your .env file or environment variables."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
