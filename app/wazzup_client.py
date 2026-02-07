from __future__ import annotations
from typing import Any, Dict
import requests

from settings import settings


class WazzupClient:
    def __init__(self) -> None:
        self.base_url = "https://api.wazzup24.com"
        self.session = requests.Session()

    def send_message(self, chat_id: str, channel_id: str, chat_type: str, text: str) -> Dict[str, Any]:
        """
        Wazzup: POST /v3/message
        """
        if not settings.WAZZUP_API_KEY:
            return {"ok": False, "error": "WAZZUP_API_KEY is empty"}

        url = f"{self.base_url}/v3/message"
        headers = {"Authorization": f"Bearer {settings.WAZZUP_API_KEY}"}
        payload = {
            "channelId": channel_id,
            "chatId": chat_id,
            "chatType": chat_type,
            "content": {"text": text},
            "type": "text",
        }
        r = self.session.post(url, json=payload, headers=headers, timeout=10)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}

        return {"ok": r.ok, "status": r.status_code, "data": data}
