from __future__ import annotations
import os
from dataclasses import dataclass


def env_str(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None else default


def env_int(key: str, default: int = 0) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = env_str("APP_NAME", "KTZH Smart Bot")

    # Mongo
    MONGO_URI: str = env_str("MONGO_URI", "mongodb+srv://barsen1506:arsen123@cluster0.ts1nh.mongodb.net/ktzh?retryWrites=true&w=majority")
    MONGO_DB: str = env_str("MONGO_DB", "ktzh")
    MONGO_SESSIONS_COLLECTION: str = env_str("MONGO_SESSIONS_COLLECTION", "sessions")
    MONGO_MESSAGES_COLLECTION: str = env_str("MONGO_MESSAGES_COLLECTION", "messages")

    # Wazzup
    WAZZUP_API_URL: str = env_str("WAZZUP_API_URL", "https://api.wazzup24.com/v3")
    WAZZUP_API_KEY: str = env_str("WAZZUP_API_KEY", "")

    # Webhook auth
    WEBHOOK_TOKEN: str = env_str("WEBHOOK_TOKEN", "")

    # Behavior
    BOT_SEND_ENABLED: bool = env_bool("BOT_SEND_ENABLED", True)
    PHONE_HASH_SALT: str = env_str("PHONE_HASH_SALT", "change_me")

    # Dialog tuning
    MEANING_MIN_SCORE: int = env_int("MEANING_MIN_SCORE", 2)
    MAX_HISTORY: int = env_int("MAX_HISTORY", 20)
    FLOOD_WINDOW_SEC: int = env_int("FLOOD_WINDOW_SEC", 8)
    FLOOD_MAX_MSG: int = env_int("FLOOD_MAX_MSG", 4)

    # Optional LLM (off by default)
    LLM_ENABLED: bool = env_bool("LLM_ENABLED", False)
    OPENAI_API_KEY: str = env_str("OPENAI_API_KEY", "")
    LLM_MODEL: str = env_str("LLM_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT_SEC: int = env_int("LLM_TIMEOUT_SEC", 6)
    LLM_STORE: bool = env_bool("LLM_STORE", False)


settings = Settings()
