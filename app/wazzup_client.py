from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class WazzupClient:
    def __init__(self, api_key: str, base_url: str = "https://api.wazzup24.com"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send_text(
        self,
        channel_id: str,
        chat_type: str,
        chat_id: str,
        text: str,
        crm_message_id: str,
    ) -> Dict[str, Any]:
        if not self._client:
            await self.start()

        payload = {
            "channelId": channel_id,
            "chatType": chat_type,
            "chatId": chat_id,
            "text": text,
            "crmMessageId": crm_message_id,
        }

        url = f"{self.base_url}/v3/message"
        r = await self._client.post(url, json=payload)
        r.raise_for_status()
        return r.json()
