"""Pydantic event schema and response models."""
import uuid
from enum import Enum
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator

class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"

class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 1
    direction: Optional[str] = None
    reentry_count: Optional[int] = None

class StoreEvent(BaseModel):
    event_id: str
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: EventType
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: EventMetadata

    @field_validator("event_id")
    @classmethod
    def validate_uuid4(cls, v: str) -> str:
        try:
            val = uuid.UUID(v, version=4)
        except ValueError:
            raise ValueError(f"'{v}' is not a valid UUID v4")
        return v

    @model_validator(mode="after")
    def validate_dwell_ms(self):
        if self.event_type == EventType.ZONE_DWELL:
            if self.dwell_ms <= 0:
                raise ValueError("dwell_ms must be > 0 for ZONE_DWELL events")
        return self

    @model_validator(mode="after")
    def validate_queue_depth(self):
        if self.event_type == EventType.BILLING_QUEUE_JOIN:
            if self.metadata.queue_depth is None or self.metadata.queue_depth <= 0:
                raise ValueError("metadata.queue_depth must be > 0 for BILLING_QUEUE_JOIN events")
        return self

class POSTransaction(BaseModel):
    store_id: str
    transaction_id: str
    timestamp: datetime
    basket_value_inr: float

class IngestRequest(BaseModel):
    events: List[StoreEvent]

    @field_validator("events")
    @classmethod
    def validate_events_length(cls, v: List[StoreEvent]) -> List[StoreEvent]:
        if not (1 <= len(v) <= 500):
            raise ValueError("events must be between 1 and 500 inclusive")
        return v

class IngestResponse(BaseModel):
    accepted: int
    rejected: int
    duplicate: int
    errors: List[dict]
