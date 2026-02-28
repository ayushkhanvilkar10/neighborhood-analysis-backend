from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class SearchCreate(BaseModel):
    neighborhood: str
    street: str
    zip_code: str


class SearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    neighborhood: str
    street: str
    zip_code: str
    created_at: datetime
    # Agent analysis fields — only present on POST, None on GET
    requests_311:    str | None = None
    crime_safety:    str | None = None
    property_mix:    str | None = None
    overall_verdict: str | None = None
