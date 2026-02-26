from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.searches import router as searches_router

app = FastAPI(title="Neighborhood Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(searches_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
