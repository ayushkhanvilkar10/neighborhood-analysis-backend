from fastapi import APIRouter, Depends, HTTPException, Response, status

from auth import get_current_user
from database import supabase
from models import SearchCreate, SearchResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/searches", response_model=SearchResponse, status_code=status.HTTP_201_CREATED)
async def create_search(search: SearchCreate, current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    supabase.postgrest.auth(token)
    row = {
        "user_id": str(user.id),
        "neighborhood": search.neighborhood,
        "street": search.street,
        "zip_code": search.zip_code,
    }
    result = (
        supabase.table("saved_searches")
        .insert(row)
        .execute()
    )
    return result.data[0]


@router.get("/searches", response_model=list[SearchResponse])
async def list_searches(current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    supabase.postgrest.auth(token)
    result = (
        supabase.table("saved_searches")
        .select("*")
        .eq("user_id", str(user.id))
        .execute()
    )
    return result.data


@router.delete("/searches/{search_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_search(search_id: str, current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    supabase.postgrest.auth(token)
    result = (
        supabase.table("saved_searches")
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