"""Shared fixtures for pytest-llm tests."""

from __future__ import annotations

import pytest

from pytest_llm.judge import LLMJudge
from pytest_llm.models import JudgeResult


class MockJudge(LLMJudge):
    """A mock LLM judge that returns configurable results without API calls."""

    def __init__(self, result: JudgeResult | None = None):
        self._mock_result = result or JudgeResult(
            passed=True, score=0.9, reason="mock", raw_response="{}"
        )
        self.provider = "mock"
        self.model = "mock"
        self.api_key = None

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        return self._mock_result


class FailingMockJudge(LLMJudge):
    """A mock LLM judge that always fails."""

    def __init__(self):
        self._mock_result = JudgeResult(
            passed=False, score=0.3, reason="mock failure", raw_response="{}"
        )
        self.provider = "mock"
        self.model = "mock"
        self.api_key = None

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        return self._mock_result


@pytest.fixture
def mock_judge():
    return MockJudge()


@pytest.fixture
def failing_mock_judge():
    return FailingMockJudge()
