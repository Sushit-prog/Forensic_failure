"""Pytest plugin for LLM-powered semantic assertions."""

from __future__ import annotations

import time
from typing import Optional

import pytest

from .config import get_configured_judge
from .judge import LLMJudge
from .reporter import LLMReporter

_reporter: Optional[LLMReporter] = None


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llm", "LLM-powered assertions")
    group.addoption(
        "--llm-judge-provider",
        choices=["openai", "anthropic", "groq", "ollama"],
        default=None,
        help="LLM provider to use for judge assertions",
    )
    group.addoption(
        "--llm-judge-model",
        default=None,
        help="Model name to use for judge assertions",
    )
    group.addoption(
        "--llm-report",
        action="store_true",
        default=False,
        help="Print a Rich-formatted summary table of LLM assertion results",
    )


@pytest.fixture(scope="session")
def llm_judge(request: pytest.FixtureRequest) -> LLMJudge:
    """Session-scoped LLMJudge instance configured from CLI or config."""
    provider = request.config.getoption("--llm-judge-provider", default=None)
    model = request.config.getoption("--llm-judge-model", default=None)

    configured = get_configured_judge()

    return LLMJudge(
        provider=provider or configured.provider,
        model=model or configured.model,
        api_key=configured.api_key,
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    global _reporter
    if config.getoption("--llm-report", default=False):
        _reporter = LLMReporter()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _reporter is None:
        return
    if report.when != "call":
        return

    if hasattr(report, "llm_assertions"):
        for assertion in report.llm_assertions:
            _reporter.record(
                assertion_name=assertion["name"],
                passed=assertion["passed"],
                score=assertion["score"],
                reason=assertion["reason"],
                test_name=report.nodeid,
            )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _reporter is not None and _reporter.results:
        _reporter.print_summary()
