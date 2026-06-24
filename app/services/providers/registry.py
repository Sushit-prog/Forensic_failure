from app.services.providers.base import BaseProvider, ProviderConfig
from app.config import get_settings


DEFAULT_MODELS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "mistral": ["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x22b"],
    "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    "openrouter": [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-2.0-flash-001",
    ],
}

COST_PER_1K = {
    "groq": {"input": 0.00059, "output": 0.00079},
    "mistral": {"input": 0.002, "output": 0.006},
    "gemini": {"input": 0.000075, "output": 0.0003},
    "openrouter": {"input": 0.00015, "output": 0.0006},
}

PROVIDER_CLASSES = {
    "groq": "app.services.providers.groq:GroqProvider",
    "mistral": "app.services.providers.mistral:MistralProvider",
    "gemini": "app.services.providers.gemini:GeminiProvider",
    "openrouter": "app.services.providers.openrouter:OpenRouterProvider",
}


def _load_provider_class(name: str):
    import importlib
    module_path, class_name = PROVIDER_CLASSES[name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_provider(name: str) -> BaseProvider:
    settings = get_settings()
    api_key_map = {
        "groq": settings.groq_api_key,
        "mistral": settings.mistral_api_key,
        "gemini": settings.gemini_api_key,
        "openrouter": settings.openrouter_api_key,
    }

    if name not in PROVIDER_CLASSES:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDER_CLASSES.keys())}")

    api_key = api_key_map.get(name, "")
    if not api_key:
        raise ValueError(f"No API key configured for provider: {name}")

    config = ProviderConfig(
        name=name,
        api_key=api_key,
        models=DEFAULT_MODELS[name],
        cost_per_1k_input=COST_PER_1K[name]["input"],
        cost_per_1k_output=COST_PER_1K[name]["output"],
    )

    cls = _load_provider_class(name)
    return cls(config)


def list_providers() -> list[dict]:
    settings = get_settings()
    api_key_map = {
        "groq": settings.groq_api_key,
        "mistral": settings.mistral_api_key,
        "gemini": settings.gemini_api_key,
        "openrouter": settings.openrouter_api_key,
    }
    return [
        {
            "name": name,
            "models": DEFAULT_MODELS[name],
            "configured": bool(api_key_map.get(name, "")),
        }
        for name in PROVIDER_CLASSES
    ]
