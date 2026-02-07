from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def env_str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = v.strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = env_str("APP_NAME", "ktzh-wazzup-bot") or "ktzh-wazzup-bot"

    # Wazzup
    WAZZUP_API_KEY: str | None = env_str("WAZZUP_API_KEY", None)
    WAZZUP_BASE_URL: str = env_str("WAZZUP_BASE_URL", "https://api.wazzup24.com") or "https://api.wazzup24.com"
    WEBHOOK_TOKEN: str | None = env_str("WEBHOOK_TOKEN", None)
    BOT_SEND_ENABLED: bool = env_bool("BOT_SEND_ENABLED", True)

    # Hashing / privacy
    PHONE_HASH_SALT: str = env_str("PHONE_HASH_SALT", "dev-salt") or "dev-salt"

    # Mongo
    MONGODB_URI: str | None = env_str("MONGODB_URI", None)
    DB_NAME: str = env_str("DB_NAME", "ktzh_bot") or "ktzh_bot"
    COL_MESSAGES: str = env_str("COL_MESSAGES", "messages") or "messages"
    COL_SESSIONS: str = env_str("COL_SESSIONS", "sessions") or "sessions"
    COL_CASES: str = env_str("COL_CASES", "cases") or "cases"

    


settings = Settings()
