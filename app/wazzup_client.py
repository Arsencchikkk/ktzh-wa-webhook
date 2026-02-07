from __future__ import annotations
from typing import Any, Dict
import httpx

from .settings import settings


class WazzupClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def send_message(self, chat_id: str, channel_id: str, chat_type: str, text: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "error": "WAZZUP_API_KEY is empty"}

        url = f"{settings.WAZZUP_API_URL.rstrip('/')}/message"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "chatId": chat_id,
            "channelId": channel_id,
            "chatType": chat_type,
            "text": text,
        }

        r = await self._client.post(url, json=payload, headers=headers)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}

        if r.status_code >= 400:
            return {"ok": False, "status": r.status_code, "response": data}

        return {"ok": True, "status": r.status_code, "response": data}
