"""LLM judge abstraction for semantic assertions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

import numpy as np

from .cache import get_cache, make_cache_key
from .models import JudgeResult

logger = logging.getLogger("pytest_llm")


_embed_model = None


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
        cache = get_cache()
        cache_key = make_cache_key(self.provider, self.model, system_prompt, user_prompt)
        cached = cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for provider=%s model=%s", self.provider, self.model)
            return cached

        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                raw = self._call_llm(system_prompt, user_prompt)
                result = self._parse_response(raw)
                cache.set(cache_key, result)
                return result
            except Exception as e:
                last_error = e
                logger.warning("Attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)

        logger.error("Judge failed after 3 attempts: %s", last_error)
        return JudgeResult(
            passed=False,
            score=0.0,
            reason=f"Judge failed after 3 attempts: {last_error}",
            raw_response="",
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()

        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        elif self.provider == "groq":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "ollama":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            response = client.chat(
                model=self.model,
                messages=messages,
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