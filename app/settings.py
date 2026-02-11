from __future__ import annotations

import os
from dataclasses import dataclass


def env_str(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None else default


def env_int(key: str, default: int = 0) -> int:
    v = os.getenv(key)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = env_str("APP_NAME", "KTZH Smart Bot")

    # Mongo
    MONGODB_URI: str = env_str("MONGODB_URI", env_str("MONGO_URI", ""))
    DB_NAME: str = env_str("MONGO_DB", "ktzh")
    COL_SESSIONS: str = env_str("MONGO_SESSIONS_COLLECTION", "sessions")
    COL_MESSAGES: str = env_str("MONGO_MESSAGES_COLLECTION", "messages")
    COL_CASES: str = env_str("MONGO_CASES_COLLECTION", "cases")
    COL_OPS_OUTBOX: str = env_str("MONGO_OPS_OUTBOX_COLLECTION", "ops_outbox")  # если вдруг оставишь

    # Wazzup
    WAZZUP_API_URL: str = env_str("WAZZUP_API_URL", "https://api.wazzup24.com/v3")
    WAZZUP_API_KEY: str = env_str("WAZZUP_API_KEY", "")

    # Webhook auth
    WEBHOOK_TOKEN: str = env_str("WEBHOOK_TOKEN", "")

    # Behavior
    BOT_SEND_ENABLED: bool = env_bool("BOT_SEND_ENABLED", True)
    PHONE_HASH_SALT: str = env_str("PHONE_HASH_SALT", "change_me")

    # Test mode
    TEST_MODE: bool = env_bool("TEST_MODE", False)
    TEST_CHAT_ID: str = env_str("TEST_CHAT_ID", "")
    TEST_CHANNEL_ID: str = env_str("TEST_CHANNEL_ID", "")

    # ✅ OPS target (куда слать оперативникам через Wazzup)
    OPS_CHANNEL_ID: str = env_str("OPS_CHANNEL_ID", "")
    OPS_CHAT_ID: str = env_str("OPS_CHAT_ID", "")
    OPS_CHAT_TYPE: str = env_str("OPS_CHAT_TYPE", "whatsapp")

    # ops_api (если хочешь оставлять ручной endpoint /api/v1/ops/send)
    OPS_SEND_TOKEN: str = env_str("OPS_SEND_TOKEN", "")


settings = Settings()
