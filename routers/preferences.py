import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Response, status
from supabase import create_client

from auth import get_current_user
from models import UserPreferencesResponse, UserPreferencesUpdate

router = APIRouter()


def get_authed_client(token: str):
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
    client.postgrest.auth(token)
    return client


@router.get("/preferences", response_model=UserPreferencesResponse)
async def get_preferences(current=Depends(get_current_user)):
    """Return the current user's saved preferences, or defaults if none exist."""
    user, token = current["user"], current["token"]
    db = get_authed_client(token)

    result = (
        db.table("user_preferences")
        .select("household_type, property_preferences, buyer_or_renter, commute_mode, interests, updated_at")
        .eq("user_id", str(user.id))
        .execute()
    )

    if result.data:
        return result.data[0]

    # No preferences saved yet — return empty defaults
    return UserPreferencesResponse()


@router.put("/preferences", response_model=UserPreferencesResponse)
async def upsert_preferences(
    prefs: UserPreferencesUpdate, current=Depends(get_current_user)
):
    """Create or update the current user's preferences (one row per user)."""
    user, token = current["user"], current["token"]
    db = get_authed_client(token)

    row = {
        "user_id":              str(user.id),
        "household_type":       prefs.household_type,
        "property_preferences": prefs.property_preferences or [],
        "buyer_or_renter":      prefs.buyer_or_renter,
        "commute_mode":         prefs.commute_mode,
        "interests":            prefs.interests or [],
        "updated_at":           datetime.now(timezone.utc).isoformat(),
    }

    result = (
        db.table("user_preferences")
        .upsert(row, on_conflict="user_id")
        .execute()
    )

    return result.data[0]


@router.delete("/preferences", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preferences(current=Depends(get_current_user)):
    """Reset preferences — deletes the user's row entirely."""
    user, token = current["user"], current["token"]
    db = get_authed_client(token)

    db.table("user_preferences").delete().eq("user_id", str(user.id)).execute()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
