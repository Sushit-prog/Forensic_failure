"""Example: Using pytest-llm assertions with a mock judge.

This example shows realistic usage of all 8 assertions.
It uses a mock judge via conftest so no real API calls are made.
To use with a real LLM, remove the mock_judge parameter from each call.
"""

from __future__ import annotations

import pytest

from pytest_llm import (
    assert_contains_claim,
    assert_faithful,
    assert_language,
    assert_no_hallucination,
    assert_regression,
    assert_safe,
    assert_semantic_similarity,
    assert_tone,
)

# Fake source document for RAG-style tests
SOURCE_DOCUMENT = """
The Python programming language was created by Guido van Rossum and first
released in 1991. Python emphasizes code readability with its use of significant
whitespace. It supports multiple programming paradigms including procedural,
object-oriented, and functional programming. Python 3.12 was released in
October 2023 and includes features like improved error messages and
performance enhancements.
"""

# Fake LLM output that summarizes the source
LLM_OUTPUT = """
Python was created by Guido van Rossum in 1991. It is known for its clean
syntax and supports object-oriented, functional, and procedural programming.
The latest version, Python 3.12, was released in October 2023 with improved
error messages and performance.
"""


class TestFaithfulAssertion:
    def test_llm_output_is_faithful_to_source(self, mock_judge):
        """The LLM output accurately reflects facts from the source document."""
        assert_faithful(LLM_OUTPUT, SOURCE_DOCUMENT, judge=mock_judge)


class TestNoHallucination:
    def test_no_hallucinated_facts(self, mock_judge):
        """Verify the output doesn't invent facts not in the source."""
        assert_no_hallucination(LLM_OUTPUT, SOURCE_DOCUMENT, judge=mock_judge)


class TestToneAssertion:
    def test_output_is_professional(self, mock_judge):
        """Check the output maintains a professional tone."""
        assert_tone(LLM_OUTPUT, "professional", judge=mock_judge)

    def test_output_is_informative(self, mock_judge):
        """Check the output has an informative tone."""
        assert_tone(LLM_OUTPUT, "informative", judge=mock_judge)


class TestSemanticSimilarity:
    def test_output_matches_expected_meaning(self):
        """The output semantically matches what we expect (no LLM needed)."""
        expected = "Python was created by Guido van Rossum in 1991 and emphasizes readability."
        assert_semantic_similarity(LLM_OUTPUT, expected, threshold=0.7)


class TestContainsClaim:
    def test_contains_creation_date(self, mock_judge):
        """Verify the output mentions Python was created in 1991."""
        assert_contains_claim(LLM_OUTPUT, "Python was created in 1991", judge=mock_judge)

    def test_contains_creator_name(self, mock_judge):
        """Verify the output mentions Guido van Rossum."""
        assert_contains_claim(LLM_OUTPUT, "Guido van Rossum created Python", judge=mock_judge)


class TestSafeAssertion:
    def test_output_is_safe(self, mock_judge):
        """Ensure the output contains no harmful content."""
        assert_safe(LLM_OUTPUT, judge=mock_judge)


class TestLanguageAssertion:
    def test_output_is_english(self, mock_judge):
        """Verify the output is written in English."""
        assert_language(LLM_OUTPUT, "en", judge=mock_judge)

    def test_output_is_english_full_name(self, mock_judge):
        """Verify the output is in English (using full language name)."""
        assert_language(LLM_OUTPUT, "English", judge=mock_judge)


class TestRegressionAssertion:
    def test_output_not_worse_than_baseline(self, mock_judge):
        """Ensure the new output isn't worse than the baseline."""
        baseline = "Python is a programming language created by Guido van Rossum in 1991."
        assert_regression(LLM_OUTPUT, baseline, threshold=0.5, judge=mock_judge)
