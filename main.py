from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.searches import router as searches_router
from routers.chat import router as chat_router
from routers.ws import router as ws_router
from routers.preferences import router as preferences_router

app = FastAPI(title="Neighborhood Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://neighborhood-analysis-frontend.vercel.app",
        "https://www.the-hunt.tech",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(searches_router)
app.include_router(chat_router)
app.include_router(ws_router)
app.include_router(preferences_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
