from sqlmodel import SQLModel, Field, Relationship
from typing import Optional
from datetime import datetime


class Trace(SQLModel, table=True):
    __tablename__ = "traces"

    id: Optional[int] = Field(default=None, primary_key=True)
    trace_id: str = Field(unique=True, index=True)
    name: Optional[str] = None
    status: str = "success"
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    metadata_json: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    spans: list["Span"] = Relationship(back_populates="trace")


class Span(SQLModel, table=True):
    __tablename__ = "spans"

    id: Optional[int] = Field(default=None, primary_key=True)
    trace_id: str = Field(foreign_key="traces.trace_id", index=True)
    span_id: str = Field(unique=True)
    parent_span_id: Optional[str] = None
    name: str
    provider: Optional[str] = None
    model: Optional[str] = None
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    status: str = "success"
    latency_ms: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error_message: Optional[str] = None
    metadata_json: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    trace: Optional[Trace] = Relationship(back_populates="spans")


class ProviderUsage(SQLModel, table=True):
    __tablename__ = "provider_usage"

    id: Optional[int] = Field(default=None, primary_key=True)
    provider: str
    model: str
    call_date: str  # YYYY-MM-DD
    call_count: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0
    estimated_cost: float = 0.0
