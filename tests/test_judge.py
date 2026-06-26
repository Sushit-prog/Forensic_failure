"""Tests for LLMJudge class."""

from __future__ import annotations

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from pytest_llm.cache import SQLiteCache, get_cache, reset_cache
from pytest_llm.judge import LLMJudge
from pytest_llm.models import JudgeResult


class TestJudgeResult:
    def test_judge_result_creation(self):
        result = JudgeResult(passed=True, score=0.85, reason="Looks good")
        assert result.passed is True
        assert result.score == 0.85
        assert result.reason == "Looks good"

    def test_judge_result_default_raw_response(self):
        result = JudgeResult(passed=False, score=0.0, reason="Failed")
        assert result.raw_response == ""


class TestLLMJudge:
    def test_default_provider_is_openai(self):
        judge = LLMJudge()
        assert judge.provider == "openai"

    def test_env_var_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_JUDGE_PROVIDER", "anthropic")
        judge = LLMJudge()
        assert judge.provider == "anthropic"

    def test_explicit_provider_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LLM_JUDGE_PROVIDER", "anthropic")
        judge = LLMJudge(provider="groq")
        assert judge.provider == "groq"

    def test_default_model_per_provider(self):
        assert LLMJudge(provider="openai").model == "gpt-4o-mini"
        assert LLMJudge(provider="anthropic").model == "claude-haiku-4-5-20251001"
        assert LLMJudge(provider="groq").model == "llama-3.3-70b-versatile"
        assert LLMJudge(provider="ollama").model == "llama3"

    def test_env_var_model(self, monkeypatch):
        monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-4o")
        judge = LLMJudge()
        assert judge.model == "gpt-4o"

    def test_parse_valid_json(self):
        judge = LLMJudge()
        raw = json.dumps({"passed": True, "score": 0.9, "reason": "All facts match"})
        result = judge._parse_response(raw)
        assert result.passed is True
        assert result.score == 0.9
        assert result.reason == "All facts match"

    def test_parse_json_with_code_block(self):
        judge = LLMJudge()
        raw = '```json\n{"passed": false, "score": 0.2, "reason": "Bad"}\n```'
        result = judge._parse_response(raw)
        assert result.passed is False
        assert result.score == 0.2

    def test_parse_malformed_json(self):
        judge = LLMJudge()
        result = judge._parse_response("this is not json")
        assert result.passed is False
        assert result.score == 0.0
        assert "unparseable" in result.reason

    def test_parse_clamps_score(self):
        judge = LLMJudge()
        raw = json.dumps({"passed": True, "score": 1.5, "reason": "Over"})
        result = judge._parse_response(raw)
        assert result.score == 1.0

    def test_parse_negative_score(self):
        judge = LLMJudge()
        raw = json.dumps({"passed": False, "score": -0.5, "reason": "Negative"})
        result = judge._parse_response(raw)
        assert result.score == 0.0

    def test_parse_missing_fields(self):
        judge = LLMJudge()
        result = judge._parse_response("{}")
        assert result.passed is False
        assert result.score == 0.0

    def test_judge_returns_result_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            reset_cache(db_path)
            judge = LLMJudge()
            mock_response = json.dumps(
                {"passed": True, "score": 0.88, "reason": "Content is faithful"}
            )
            with patch.object(judge, "_call_llm", return_value=mock_response):
                result = judge.judge("system", "user")
                assert result.passed is True
                assert result.score == 0.88

    def test_judge_retries_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            reset_cache(db_path)
            judge = LLMJudge()
            call_count = 0

            def mock_call(system_prompt, user_prompt):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("API error")
                return json.dumps({"passed": True, "score": 0.75, "reason": "Retry worked"})

            with patch.object(judge, "_call_llm", side_effect=mock_call):
                with patch("pytest_llm.judge.time.sleep") as mock_sleep:
                    result = judge.judge("system", "user")
                    assert result.passed is True
                    assert call_count == 3
                    assert mock_sleep.call_count == 2
                    mock_sleep.assert_any_call(1)
                    mock_sleep.assert_any_call(2)

    def test_judge_returns_failure_after_max_retries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            reset_cache(db_path)
            judge = LLMJudge()

            def mock_call(system_prompt, user_prompt):
                raise Exception("API error")

            with patch.object(judge, "_call_llm", side_effect=mock_call):
                result = judge.judge("system", "user")
                assert result.passed is False
                assert "failed after 3 attempts" in result.reason


class TestLLMJudgeEmbed:
    def test_embed_returns_vector(self):
        judge = LLMJudge()
        vec = judge.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_consistency(self):
        judge = LLMJudge()
        vec1 = judge.embed("test sentence")
        vec2 = judge.embed("test sentence")
        assert vec1 == vec2


class TestJudgeCache:
    def test_judge_cache_hit_avoids_api_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            reset_cache(db_path)
            judge = LLMJudge(provider="openai", model="gpt-4o-mini")
            mock_response = json.dumps(
                {"passed": True, "score": 0.9, "reason": "Cached result"}
            )
            with patch.object(judge, "_call_llm", return_value=mock_response) as mock_call:
                result1 = judge.judge("system", "user")
                result2 = judge.judge("system", "user")
                assert mock_call.call_count == 1
                assert result1.passed is True
                assert result2.passed is True
                assert result1.score == result2.score

    def test_cache_ttl_expiry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            cache = SQLiteCache(db_path)
            result = JudgeResult(passed=True, score=0.8, reason="test")
            cache.set("test_key", result)

            retrieved = cache.get("test_key")
            assert retrieved is not None
            assert retrieved.passed is True

            with patch("pytest_llm.cache.time.time", return_value=time.time() + 8 * 24 * 60 * 60):
                retrieved = cache.get("test_key")
                assert retrieved is None

    def test_cache_eviction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            cache = SQLiteCache(db_path)
            conn = cache._conn()
            try:
                for i in range(10001):
                    result = JudgeResult(passed=True, score=0.5, reason=f"test {i}")
                    conn.execute(
                        "INSERT OR REPLACE INTO judge_cache (key, result_json, created_at) VALUES (?, ?, ?)",
                        (f"key_{i}", json.dumps(result.model_dump()), float(i)),
                    )
                conn.commit()
                cache._evict_if_needed(conn)
                count = conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0]
                assert count <= 10000
            finally:
                conn.close()