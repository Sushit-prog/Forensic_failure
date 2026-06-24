from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None
    raw: Optional[dict] = None


class ProviderConfig(BaseModel):
    name: str
    api_key: str
    models: list[str]
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 4096
    supports_streaming: bool = True


class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        pass

    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (
            (tokens_in / 1000) * self.config.cost_per_1k_input
            + (tokens_out / 1000) * self.config.cost_per_1k_output
        )

    def get_default_model(self) -> str:
        return self.config.models[0] if self.config.models else "unknown"
