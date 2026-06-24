from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./data/reliability.db"
    app_env: str = "development"
    secret_key: str = "change-me-in-production"

    groq_api_key: str = ""
    mistral_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
