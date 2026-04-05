from fastapi import APIRouter, Depends, HTTPException, Response, status
import json
import os
from pathlib import Path
from supabase import create_client

from auth import get_current_user
from models import SearchCreate, SearchResponse
from agent.neighborhood_analysis import graph

# Load neighborhood tiers once at startup
_tiers_path = Path(__file__).resolve().parent.parent / "agent" / "neighborhood_tiers.json"
with open(_tiers_path) as _f:
    NEIGHBORHOOD_TIERS = json.load(_f)

router = APIRouter()


def get_authed_client(token: str):
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
    client.postgrest.auth(token)
    return client


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/searches", response_model=SearchResponse, status_code=status.HTTP_201_CREATED)
async def create_search(search: SearchCreate, current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    db = get_authed_client(token)
    # Run the agent first — street_name maps to the street field
    agent_result = await graph.ainvoke({
        "neighborhood":         search.neighborhood,
        "street_name":          search.street,
        "zip_code":             search.zip_code,
        "household_type":       search.household_type,
        "property_preferences": search.property_preferences,
        "buyer_or_renter":      search.buyer_or_renter,
        "commute_mode":         search.commute_mode,
        "interests":            search.interests,
    })

    analysis_dict = {
        "requests_311":           agent_result["requests_311"],
        "crime_safety":           agent_result["crime_safety"],
        "property_mix":           agent_result["property_mix"],
        "permit_activity":        agent_result["permit_activity"],
        "entertainment_scene":    agent_result["entertainment_scene"],
        "traffic_safety":         agent_result["traffic_safety"],
        "gun_violence":           agent_result["gun_violence"],
        "green_space":            agent_result["green_space"],
        "overall_verdict":        agent_result["overall_verdict"],
        "closing_recommendation": agent_result["closing_recommendation"],
        "raw_stats":              agent_result.get("raw_stats", []),
        "neighborhood_tiers":     NEIGHBORHOOD_TIERS.get(search.neighborhood),
    }

    # Insert row with analysis already populated
    row = {
        "user_id":      str(user.id),
        "neighborhood": search.neighborhood,
        "street":       search.street,
        "zip_code":     search.zip_code,
        "analysis":     analysis_dict,
    }
    result = db.table("saved_searches").insert(row).execute()
    saved = result.data[0]

    return {**saved, "analysis": analysis_dict}


@router.get("/searches", response_model=list[SearchResponse])
async def list_searches(current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    db = get_authed_client(token)
    result = (
        db.table("saved_searches")
        .select("*")
        .eq("user_id", str(user.id))
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


@router.delete("/searches/{search_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_search(search_id: str, current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    db = get_authed_client(token)
    result = (
        db.table("saved_searches")
        .delete()
        .eq("id", search_id)
        .eq("user_id", str(user.id))
        .execute()
    )
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Search not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
