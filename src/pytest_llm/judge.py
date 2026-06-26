"""LLM judge abstraction for semantic assertions."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel


class JudgeResult(BaseModel):
    """Result from an LLM judge evaluation."""

    passed: bool
    score: float
    reason: str
    raw_response: str = ""


_embed_model = None


class JudgeCache:
    def __init__(self):
        self._cache: dict[str, JudgeResult] = {}

    def _key(self, provider: str, model: str, system_prompt: str, user_prompt: str) -> str:
        content = f"{provider}:{model}:{system_prompt}:{user_prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, provider: str, model: str, system_prompt: str, user_prompt: str) -> JudgeResult | None:
        return self._cache.get(self._key(provider, model, system_prompt, user_prompt))

    def set(self, provider: str, model: str, system_prompt: str, user_prompt: str, result: JudgeResult) -> None:
        self._cache[self._key(provider, model, system_prompt, user_prompt)] = result


_cache_instance = JudgeCache()


class LLMJudge:
    """
    Wraps any LLM provider. Called by all assertion functions.

    Provider selected by env var LLM_JUDGE_PROVIDER or fixture override.
    Supported providers: openai, anthropic, groq, ollama
    Default provider: openai
    Default model per provider:
        openai -> gpt-4o-mini
        anthropic -> claude-haiku-4-5-20251001
        groq -> llama-3.3-70b-versatile
        ollama -> llama3
    """

    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "groq": "llama-3.3-70b-versatile",
        "ollama": "llama3",
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider or os.environ.get("LLM_JUDGE_PROVIDER", "openai")
        self.model = model or os.environ.get("LLM_JUDGE_MODEL") or self.DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self.api_key = api_key or self._resolve_api_key()
        self._client: Any = None

    def _resolve_api_key(self) -> Optional[str]:
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "ollama": None,
        }
        env_var = env_map.get(self.provider)
        if env_var:
            return os.environ.get(env_var)
        return None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            import openai

            self._client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "groq":
            import groq

            self._client = groq.Groq(api_key=self.api_key)
        elif self.provider == "ollama":
            import ollama as ollama_mod

            self._client = ollama_mod
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return self._client

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        """
        Returns JudgeResult with:
            passed: bool
            score: float (0.0 to 1.0)
            reason: str (one sentence explanation)
            raw_response: str
        """
        cached = _cache_instance.get(self.provider, self.model, system_prompt, user_prompt)
        if cached:
            return cached

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                raw = self._call_llm(full_prompt)
                result = self._parse_response(raw)
                _cache_instance.set(self.provider, self.model, system_prompt, user_prompt, result)
                return result
            except Exception as e:
                last_error = e
                if attempt < 2:
                    continue

        return JudgeResult(
            passed=False,
            score=0.0,
            reason=f"Judge failed after 3 attempts: {last_error}",
            raw_response="",
        )

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()

        if self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.provider == "groq":
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "ollama":
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _parse_response(self, raw: str) -> JudgeResult:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return JudgeResult(
                passed=False,
                score=0.0,
                reason="Judge returned unparseable response",
                raw_response=raw,
            )

        passed = bool(data.get("passed", False))
        score = float(data.get("score", 0.0))
        reason = str(data.get("reason", ""))

        return JudgeResult(
            passed=passed,
            score=max(0.0, min(1.0, score)),
            reason=reason,
            raw_response=raw,
        )

    def embed(self, text: str) -> list[float]:
        """
        Returns embedding vector using sentence-transformers all-MiniLM-L6-v2.
        Used for semantic similarity assertions.
        Always local, never calls an API.
        """
        global _embed_model
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer

            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _embed_model.encode(text).tolist()
