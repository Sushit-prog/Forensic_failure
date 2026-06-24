import time
import httpx
from app.services.providers.base import BaseProvider, LLMResponse, ProviderConfig


class OpenRouterProvider(BaseProvider):
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=60.0,
        )

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        model = model or self.get_default_model()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = await self.client.post(
            "/chat/completions",
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        )
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=model,
            provider="openrouter",
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def close(self):
        await self.client.aclose()
