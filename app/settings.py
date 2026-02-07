import os
import json
from typing import Any, Dict, Optional

def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _get_json(name: str, default: Any) -> Any:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return json.loads(v)
    except Exception:
        return default


LLM_ENABLED = env_bool("LLM_ENABLED", False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "6"))
LLM_STORE = env_bool("LLM_STORE", False)  # Рекомендовано false
LLM_CONFIDENCE_THRESHOLD = int(os.getenv("LLM_CONFIDENCE_THRESHOLD", "60"))  # 0..100
LLM_MAX_HISTORY = int(os.getenv("LLM_MAX_HISTORY", "12"))
LLM_MAX_TEXT_CHARS = int(os.getenv("LLM_MAX_TEXT_CHARS", "800"))


# Mongo
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
DB_NAME = os.getenv("DB_NAME", "ktzh").strip()
COL_MESSAGES = os.getenv("COL_MESSAGES", "messages").strip()
COL_SESSIONS = os.getenv("COL_SESSIONS", "sessions").strip()
COL_CASES = os.getenv("COL_CASES", "cases").strip()


# Webhook security
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
TEST_ONLY_CHAT_ID = os.getenv("TEST_ONLY_CHAT_ID", "").strip()

# Optional phone hashing
PHONE_HASH_SALT = os.getenv("PHONE_HASH_SALT", "").strip()

# Wazzup API
WAZZUP_API_KEY = os.getenv("WAZZUP_API_KEY", "").strip()
BOT_SEND_ENABLED = _get_bool("BOT_SEND_ENABLED", default=True)

# Routing (replace with your real executors later)
# Example:
# EXECUTORS_JSON='{"center":{"chatType":"whatsapp","chatId":"77001112233"}, "east":{...}}'
EXECUTORS: Dict[str, Dict[str, str]] = _get_json("EXECUTORS_JSON", default={})

# Lost&Found target (group or phone)
# Example:
# LOST_FOUND_TARGET_JSON='{"chatType":"whatsgroup","chatId":"<group_chat_id_from_webhook>"}'
LOST_FOUND_TARGET: Dict[str, str] = _get_json("LOST_FOUND_TARGET_JSON", default={})

# Support/Operator target for info requests
SUPPORT_TARGET: Dict[str, str] = _get_json("SUPPORT_TARGET_JSON", default={})

# Routing rules (optional)
# Example:
# ROUTING_RULES_JSON='[{"match":{"train_regex":"^T58$"},"region":"east"}]'
ROUTING_RULES = _get_json("ROUTING_RULES_JSON", default=[])
# ===== TEST MODE (only your number) =====
TEST_CHANNEL_ID = os.getenv("TEST_CHANNEL_ID", "92f607bd-a68b-44c2-b718-aed344d72435").strip()
TEST_CHAT_ID = os.getenv("TEST_CHAT_ID", "77052817121").strip()
TEST_CHAT_TYPE = os.getenv("TEST_CHAT_TYPE", "whatsapp").strip()

# Ответы бота только этим chatId (через запятую): "77052817121,7777...."
_allowed = os.getenv("ALLOWED_CHAT_IDS", "77052817121").strip()
ALLOWED_CHAT_IDS = [x.strip() for x in _allowed.split(",") if x.strip()] if _allowed else []
