from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


# ─────────────────────────────────────────────
# Searches
# ─────────────────────────────────────────────

class SearchCreate(BaseModel):
    neighborhood:         str
    street:               str
    zip_code:             str
    household_type:       str | None = None
    property_preferences: list[str] | None = None
    buyer_or_renter:      str | None = None
    commute_mode:         str | None = None
    interests:            list[str] | None = None

    @field_validator("property_preferences")
    @classmethod
    def max_two(cls, v: list[str] | None) -> list[str] | None:
        if v and len(v) > 2:
            raise ValueError("Maximum 2 property preferences allowed")
        return v


class SearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:           UUID
    user_id:      UUID
    neighborhood: str
    street:       str
    zip_code:     str
    created_at:   datetime
    # Agent analysis — stored as JSONB, only present on POST, None on GET
    analysis:     dict | None = None


# ─────────────────────────────────────────────
# Chat sessions
# ─────────────────────────────────────────────

class ChatSessionCreate(BaseModel):
    first_message: str  # Used to create the session title and kick off the first turn


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:         UUID
    title:      str
    created_at: datetime


# ─────────────────────────────────────────────
# Chat messages
# ─────────────────────────────────────────────

class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:         UUID
    role:       str   # 'human' | 'ai'
    content:    str
    created_at: datetime


# ─────────────────────────────────────────────
# User preferences
# ─────────────────────────────────────────────

HouseholdType = Literal[
    "Living solo",
    "Couple / Partner",
    "Family with kids",
    "Retiree / Empty nester",
    "Investor",
]

PropertyPreference = Literal[
    "Condo",
    "Single Family",
    "Two / Three Family",
    "Small Apartment",
    "Mid-Size Apartment",
    "Mixed Use",
]

BuyerOrRenter = Literal["Buyer", "Renter"]

CommuteMode = Literal[
    "Car",
    "Public transit",
    "Bike",
    "Walk",
    "Remote / No commute",
]

Interest = Literal[
    "eat out",
    "go out for drinks",
    "grab coffee",
    "attend live events",
    "browse local shops",
    "run & cycle",
    "go for walks",
    "walk my dog",
    "explore parks & nature",
    "garden",
    "cook at home",
    "order takeout",
    "watch TV",
]


class UserPreferencesUpdate(BaseModel):
    """Request body for PUT /preferences. All fields optional."""
    household_type:        HouseholdType | None = None
    property_preferences:  list[PropertyPreference] | None = None
    buyer_or_renter:       BuyerOrRenter | None = None
    commute_mode:          CommuteMode | None = None
    interests:             list[Interest] | None = None
    onboarding_completed:  bool | None = None

    @field_validator("property_preferences")
    @classmethod
    def max_two(cls, v: list[str] | None) -> list[str] | None:
        if v and len(v) > 2:
            raise ValueError("Maximum 2 property preferences allowed")
        return v


class UserPreferencesResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    household_type:        str | None = None
    property_preferences:  list[str] | None = None
    buyer_or_renter:       str | None = None
    commute_mode:          str | None = None
    interests:             list[str] | None = None
    onboarding_completed:  bool | None = None
    updated_at:            datetime | None = None
