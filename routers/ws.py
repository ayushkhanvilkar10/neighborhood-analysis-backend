import os
from supabase import create_client

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from langchain_core.messages import HumanMessage, AIMessage

from database import supabase
from agent.chat_agent import stream_chat

router = APIRouter(tags=["websocket"])


def get_authed_client(token: str):
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
    client.postgrest.auth(token)
    return client


async def authenticate_ws(token: str) -> dict | None:
    """Verify the JWT token and return the user, or None if invalid."""
    try:
        response = supabase.auth.get_user(token)
        return response.user
    except Exception:
        return None


async def load_history(db, session_id: str) -> list:
    """
    Fetch all prior messages for a session from Supabase and reconstruct
    them as LangChain message objects for passing into the chat agent.
    """
    result = (
        db.table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    messages = []
    for row in result.data:
        if row["role"] == "human":
            messages.append(HumanMessage(content=row["content"]))
        else:
            messages.append(AIMessage(content=row["content"]))
    return messages


async def save_message(db, session_id: str, user_id: str, role: str, content: str):
    """Persist a single message to the chat_messages table."""
    db.table("chat_messages").insert({
        "session_id": session_id,
        "user_id":    user_id,
        "role":       role,
        "content":    content,
    }).execute()


# ─────────────────────────────────────────────
# WebSocket /ws/chat/{session_id}
# ─────────────────────────────────────────────

@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),  # JWT passed as ?token=... query param
):
    # 1. Authenticate before accepting the connection
    user = await authenticate_ws(token)
    if not user:
        await websocket.close(code=4001)
        return

    db = get_authed_client(token)

    # 2. Verify the session belongs to this user
    session = (
        db.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", str(user.id))
        .execute()
    )
    if not session.data:
        await websocket.close(code=4004)
        return

    await websocket.accept()

    try:
        while True:
            # 3. Wait for the next user message
            user_message = await websocket.receive_text()
            if not user_message.strip():
                continue

            # 4. Save the human message to Supabase
            await save_message(db, session_id, str(user.id), "human", user_message)

            # 5. Load full conversation history from Supabase
            #    (includes the message we just saved)
            history = await load_history(db, session_id)

            # 6. Stream tokens back to the client as they arrive
            full_response = ""
            async for token_chunk in stream_chat(history, config={}):
                await websocket.send_text(token_chunk)
                full_response += token_chunk

            # 7. Send an end-of-stream signal so the client knows the turn is complete
            await websocket.send_text("[DONE]")

            # 8. Save the complete AI response to Supabase
            await save_message(db, session_id, str(user.id), "ai", full_response)

    except WebSocketDisconnect:
        pass  # Client disconnected cleanly — nothing to do
