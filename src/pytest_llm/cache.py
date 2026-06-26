"""SQLite-backed cache for LLM judge results."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Optional

from .models import JudgeResult

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pytest_llm")
DEFAULT_CACHE_DB = os.path.join(DEFAULT_CACHE_DIR, "cache.db")
TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
MAX_ROWS = 10000
DELETE_BATCH = 1000


class SQLiteCache:
    """SQLite-backed cache with TTL and max size enforcement."""

    def __init__(self, db_path: str = DEFAULT_CACHE_DB):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get(self, key: str) -> Optional[JudgeResult]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT result_json, created_at FROM judge_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            result_json, created_at = row
            if time.time() - created_at > TTL_SECONDS:
                conn.execute("DELETE FROM judge_cache WHERE key = ?", (key,))
                conn.commit()
                return None
            data = json.loads(result_json)
            return JudgeResult(
                passed=data["passed"],
                score=data["score"],
                reason=data["reason"],
                raw_response=data.get("raw_response", ""),
            )
        finally:
            conn.close()

    def set(self, key: str, result: JudgeResult) -> None:
        conn = self._conn()
        try:
            result_json = json.dumps(result.model_dump())
            conn.execute(
                "INSERT OR REPLACE INTO judge_cache (key, result_json, created_at) VALUES (?, ?, ?)",
                (key, result_json, time.time()),
            )
            conn.commit()
            self._evict_if_needed(conn)
        finally:
            conn.close()

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        count = conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0]
        if count > MAX_ROWS:
            conn.execute(
                f"""
                DELETE FROM judge_cache WHERE key IN (
                    SELECT key FROM judge_cache ORDER BY created_at ASC LIMIT {DELETE_BATCH}
                )
                """
            )
            conn.commit()

    def clear(self) -> None:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM judge_cache")
            conn.commit()
        finally:
            conn.close()


def make_cache_key(provider: str, model: str, system_prompt: str, user_prompt: str) -> str:
    import hashlib
    content = f"{provider}:{model}:{system_prompt}:{user_prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


_cache_instance: Optional[SQLiteCache] = None


def get_cache() -> SQLiteCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SQLiteCache()
    return _cache_instance


def reset_cache(db_path: str = DEFAULT_CACHE_DB) -> SQLiteCache:
    global _cache_instance
    _cache_instance = SQLiteCache(db_path)
    return _cache_instance