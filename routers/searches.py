from fastapi import APIRouter, Depends, HTTPException, Response, status
import os
from supabase import create_client

from auth import get_current_user
from models import SearchCreate, SearchResponse
from agent.neighborhood_analysis import graph

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
    row = {
        "user_id": str(user.id),
        "neighborhood": search.neighborhood,
        "street": search.street,
        "zip_code": search.zip_code,
    }
    result = db.table("saved_searches").insert(row).execute()
    saved = result.data[0]

    # Run the agent — street_name maps to the street field
    analysis = await graph.ainvoke({
        "neighborhood": search.neighborhood,
        "street_name":  search.street,
        "zip_code":     search.zip_code,
    })

    return {
        **saved,
        "requests_311":    analysis["requests_311"],
        "crime_safety":    analysis["crime_safety"],
        "property_mix":    analysis["property_mix"],
        "permit_activity":     analysis["permit_activity"],
        "entertainment_scene": analysis["entertainment_scene"],
        "traffic_safety":      analysis["traffic_safety"],
        "gun_violence":        analysis["gun_violence"],
        "green_space":         analysis["green_space"],
        "overall_verdict":     analysis["overall_verdict"],
    }


@router.get("/searches", response_model=list[SearchResponse])
async def list_searches(current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    db = get_authed_client(token)
    result = (
        db.table("saved_searches")
        .select("*")
        .eq("user_id", str(user.id))
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
