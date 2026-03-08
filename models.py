from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


class SearchCreate(BaseModel):
    neighborhood:        str
    street:              str
    zip_code:            str
    household_type:      str | None = None
    property_preferences: list[str] | None = None

    @field_validator("property_preferences")
    @classmethod
    def max_two(cls, v: list[str] | None) -> list[str] | None:
        if v and len(v) > 2:
            raise ValueError("Maximum 2 property preferences allowed")
        return v


class SearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    neighborhood: str
    street: str
    zip_code: str
    created_at: datetime
    # Agent analysis — stored as JSONB, only present on POST, None on GET
    analysis: dict | None = None
