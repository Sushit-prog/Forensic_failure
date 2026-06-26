"""Configuration helpers for pytest-llm."""

from __future__ import annotations

import logging
from typing import Optional

from .judge import LLMJudge

logger = logging.getLogger("pytest_llm")

_config: dict = {}


def pytest_configure_judge(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Call this from your conftest.py to configure the judge globally:

        # conftest.py
        from pytest_llm import pytest_configure_judge
        pytest_configure_judge(provider="anthropic", model="claude-haiku-4-5-20251001")
    """
    if provider is not None:
        _config["provider"] = provider
    if model is not None:
        _config["model"] = model
    if api_key is not None:
        _config["api_key"] = api_key


def get_configured_judge() -> LLMJudge:
    """Create an LLMJudge from the current configuration."""
    return LLMJudge(
        provider=_config.get("provider"),
        model=_config.get("model"),
        api_key=_config.get("api_key"),
    )
