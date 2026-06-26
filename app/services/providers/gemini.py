import time
import google.generativeai as genai
from app.services.providers.base import BaseProvider, LLMResponse, ProviderConfig


class GeminiProvider(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.client = genai.GenerativeModel(model_name=config.models[0] if config.models else "gemini-1.5-flash")

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        model_name = model or self.get_default_model()
        gen_model = genai.GenerativeModel(model_name=model_name)

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        start = time.perf_counter()
        response = await gen_model.generate_content_async(
            full_prompt,
            generation_config=genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        tokens_in = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        tokens_out = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

        return LLMResponse(
            content=response.text or "",
            model=model_name,
            provider="gemini",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw={"text": response.text, "candidates": [c.__dict__ for c in response.candidates] if response.candidates else []},
        )
