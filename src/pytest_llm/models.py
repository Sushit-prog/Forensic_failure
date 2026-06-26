"""Shared models for pytest-llm."""

from __future__ import annotations

from pydantic import BaseModel


class JudgeResult(BaseModel):
    """Result from an LLM judge evaluation."""

    passed: bool
    score: float
    reason: str
    raw_response: str = ""