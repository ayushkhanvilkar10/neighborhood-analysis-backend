from fastapi import APIRouter, Depends, HTTPException, status
import os
from supabase import create_client

from auth import get_current_user
from models import ChatSessionCreate, ChatSessionResponse, ChatMessageResponse

router = APIRouter(prefix="/chat", tags=["chat"])

TITLE_MAX_LENGTH = 60


def get_authed_client(token: str):
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
    client.postgrest.auth(token)
    return client


# ─────────────────────────────────────────────
# POST /chat/sessions
# Create a new chat session. Title is derived from the first message.
# ─────────────────────────────────────────────

@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(body: ChatSessionCreate, current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    db = get_authed_client(token)

    title = body.first_message.strip()[:TITLE_MAX_LENGTH]

    result = db.table("chat_sessions").insert({
        "user_id": str(user.id),
        "title":   title,
    }).execute()

    return result.data[0]


# ─────────────────────────────────────────────
# GET /chat/sessions
# List all chat sessions for the current user, newest first.
# ─────────────────────────────────────────────

@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    db = get_authed_client(token)

    result = (
        db.table("chat_sessions")
        .select("*")
        .eq("user_id", str(user.id))
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


# ─────────────────────────────────────────────
# GET /chat/sessions/{session_id}/messages
# Load full message history for a session, oldest first (chat order).
# ─────────────────────────────────────────────

@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def list_messages(session_id: str, current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    db = get_authed_client(token)

    # Verify the session belongs to the current user
    session = (
        db.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", str(user.id))
        .execute()
    )
    if not session.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    result = (
        db.table("chat_messages")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data


# ─────────────────────────────────────────────
# DELETE /chat/sessions/{session_id}
# Delete a session and all its messages (cascade handled by DB).
# ─────────────────────────────────────────────

@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, current=Depends(get_current_user)):
    user, token = current["user"], current["token"]
    db = get_authed_client(token)

    result = (
        db.table("chat_sessions")
        .delete()
        .eq("id", session_id)
        .eq("user_id", str(user.id))
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
