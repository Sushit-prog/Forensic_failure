"""Tests for LLM-powered assertion functions."""

from __future__ import annotations

import numpy as np
import pytest

from pytest_llm.assertions import (
    assert_contains_claim,
    assert_faithful,
    assert_language,
    assert_no_hallucination,
    assert_regression,
    assert_safe,
    assert_semantic_similarity,
    assert_tone,
)
from pytest_llm.judge import JudgeResult


class TestAssertFaithful:
    def test_faithful_output_passes(self, mock_judge):
        output = "The cat sat on the mat."
        source = "There was a cat. It sat on the mat."
        assert_faithful(output, source, judge=mock_judge)

    def test_unfaithful_output_fails(self, failing_mock_judge):
        output = "The dog ran in the park."
        source = "The cat sat on the mat."
        with pytest.raises(AssertionError, match="assert_faithful failed"):
            assert_faithful(output, source, judge=failing_mock_judge)

    def test_faithful_with_custom_threshold(self, mock_judge):
        output = "The cat sat on the mat."
        source = "There was a cat. It sat on the mat."
        assert_faithful(output, source, threshold=0.5, judge=mock_judge)


class TestAssertNoHallucination:
    def test_no_hallucination_passes(self, mock_judge):
        output = "The sky is blue."
        source = "The sky appears blue during the day."
        assert_no_hallucination(output, source, judge=mock_judge)

    def test_hallucination_detected(self, failing_mock_judge):
        output = "The sky is blue and has 7 moons."
        source = "The sky appears blue during the day."
        with pytest.raises(AssertionError, match="assert_no_hallucination failed"):
            assert_no_hallucination(output, source, judge=failing_mock_judge)


class TestAssertTone:
    def test_professional_tone_passes(self, mock_judge):
        output = "We would like to inform you that your request has been processed."
        assert_tone(output, "professional", judge=mock_judge)

    def test_casual_vs_formal_fails(self, failing_mock_judge):
        output = "Yo dude, check this out, it's pretty dope."
        with pytest.raises(AssertionError, match="assert_tone failed"):
            assert_tone(output, "formal", judge=failing_mock_judge)

    def test_tone_with_custom_threshold(self, mock_judge):
        output = "Thank you for your inquiry."
        assert_tone(output, "polite", threshold=0.5, judge=mock_judge)


class TestAssertSemanticSimilarity:
    def test_high_similarity_passes(self):
        output = "The quick brown fox jumps over the lazy dog."
        expected = "A fast brown fox leaps over an idle dog."
        assert_semantic_similarity(output, expected, threshold=0.8)

    def test_low_similarity_fails(self):
        output = "I love programming in Python."
        expected = "Quantum physics describes subatomic particles."
        with pytest.raises(AssertionError, match="assert_semantic_similarity failed"):
            assert_semantic_similarity(output, expected, threshold=0.8)

    def test_identical_strings_score_near_one(self):
        text = "Machine learning is a subset of artificial intelligence."
        j = __import__("pytest_llm.judge", fromlist=["LLMJudge"]).LLMJudge()
        vec = np.array(j.embed(text))
        similarity = float(np.dot(vec, vec) / (np.linalg.norm(vec) * np.linalg.norm(vec)))
        assert similarity > 0.99


class TestAssertContainsClaim:
    def test_claim_present_passes(self, mock_judge):
        output = "Python was created by Guido van Rossum in 1991."
        assert_contains_claim(output, "Python was created in the 1990s", judge=mock_judge)

    def test_claim_absent_fails(self, failing_mock_judge):
        output = "Python is a programming language."
        with pytest.raises(AssertionError, match="assert_contains_claim failed"):
            assert_contains_claim(output, "Python was created in 1989", judge=failing_mock_judge)


class TestAssertSafe:
    def test_safe_content_passes(self, mock_judge):
        output = "The weather is nice today and I enjoy reading books."
        assert_safe(output, judge=mock_judge)

    def test_unsafe_content_fails(self, failing_mock_judge):
        output = "This contains harmful instructions."
        with pytest.raises(AssertionError, match="assert_safe failed"):
            assert_safe(output, judge=failing_mock_judge)


class TestAssertLanguage:
    def test_correct_language_passes(self, mock_judge):
        output = "The quick brown fox jumps over the lazy dog."
        assert_language(output, "en", judge=mock_judge)

    def test_wrong_language_fails(self, failing_mock_judge):
        output = "The quick brown fox jumps over the lazy dog."
        with pytest.raises(AssertionError, match="assert_language failed"):
            assert_language(output, "French", judge=failing_mock_judge)


class TestAssertRegression:
    def test_no_regression_passes(self, mock_judge):
        output = "Python is a versatile programming language."
        baseline = "Python is a programming language used for many purposes."
        assert_regression(output, baseline, judge=mock_judge)

    def test_content_drift_fails(self):
        output = "Quantum computing uses qubits for parallel processing."
        baseline = "Python is a versatile programming language."
        with pytest.raises(AssertionError, match="content drift"):
            assert_regression(output, baseline, threshold=0.85)

    def test_quality_regression_fails(self, failing_mock_judge):
        output = "bad"
        baseline = "good"
        with pytest.raises(AssertionError, match="assert_regression failed"):
            assert_regression(output, baseline, threshold=0.0, judge=failing_mock_judge)
