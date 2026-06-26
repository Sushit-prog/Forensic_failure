"""Rich-based reporter for LLM assertion results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from rich.console import Console
from rich.table import Table

logger = logging.getLogger("pytest_llm")


@dataclass
class AssertionResult:
    """Result of a single LLM assertion."""

    assertion_name: str
    test_name: str
    passed: bool
    score: float
    reason: str
    duration_ms: float = 0.0


class LLMReporter:
    """Collects and displays LLM assertion results."""

    def __init__(self) -> None:
        self.results: List[AssertionResult] = []

    def record(
        self,
        assertion_name: str,
        passed: bool,
        score: float,
        reason: str,
        test_name: str,
    ) -> None:
        self.results.append(
            AssertionResult(
                assertion_name=assertion_name,
                test_name=test_name,
                passed=passed,
                score=score,
                reason=reason,
            )
        )

    def print_summary(self) -> None:
        console = Console()
        table = Table(title="LLM Assertion Results", show_lines=True)
        table.add_column("Test", style="bold")
        table.add_column("Assertion")
        table.add_column("Passed")
        table.add_column("Score")
        table.add_column("Reason")

        passed_count = 0
        failed_count = 0

        for r in self.results:
            status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            score_str = f"{r.score:.2f}"
            reason_display = r.reason[:60] + "..." if len(r.reason) > 60 else r.reason

            if r.passed:
                passed_count += 1
            else:
                failed_count += 1

            table.add_row(
                r.test_name,
                r.assertion_name,
                status,
                score_str,
                reason_display,
            )

        console.print(table)
        console.print(
            f"\n[bold]Total:[/bold] {passed_count} passed, {failed_count} failed, "
            f"{len(self.results)} assertions"
        )
