"""Shared fixtures for example tests."""

from __future__ import annotations

import pytest

from pytest_llm.judge import JudgeResult, LLMJudge


class MockJudge(LLMJudge):
    def __init__(self):
        self._mock_result = JudgeResult(
            passed=True, score=0.9, reason="mock", raw_response="{}"
        )
        self.provider = "mock"
        self.model = "mock"
        self.api_key = None

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        return self._mock_result


@pytest.fixture
def mock_judge():
    return MockJudge()
