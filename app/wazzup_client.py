from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any

import httpx

from . import settings

@dataclass
class SendResult:
    ok: bool
    response: Dict[str, Any]

class WazzupClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.wazzup24.com/v3",
                timeout=httpx.Timeout(20.0)
            )

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send_text(self, *, channel_id: str, chat_type: str, chat_id: str, text: str, crm_message_id: str) -> SendResult:
        return await self._send(
            payload={
                "channelId": channel_id,
                "chatType": chat_type,
                "chatId": chat_id,
                "text": text,
                "crmMessageId": crm_message_id,
            }
        )

    async def send_file(self, *, channel_id: str, chat_type: str, chat_id: str, content_uri: str, crm_message_id: str) -> SendResult:
        return await self._send(
            payload={
                "channelId": channel_id,
                "chatType": chat_type,
                "chatId": chat_id,
                "contentUri": content_uri,
                "crmMessageId": crm_message_id,
            }
        )

    async def _send(self, payload: Dict[str, Any]) -> SendResult:
        if not settings.BOT_SEND_ENABLED:
            return SendResult(ok=True, response={"dryRun": True, "payload": payload})

        if not self.api_key:
            return SendResult(ok=False, response={"error": "WAZZUP_API_KEY is empty"})

        await self.start()
        assert self._client is not None

        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Retry on 429 with small backoff
        for attempt in range(4):
            r = await self._client.post("/message", headers=headers, json=payload)
            if r.status_code == 429:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            if 200 <= r.status_code < 300:
                data = r.json() if r.content else {"ok": True}
                return SendResult(ok=True, response=data)
            # other errors
            try:
                data = r.json()
            except Exception:
                data = {"status_code": r.status_code, "text": r.text}
            return SendResult(ok=False, response=data)

        return SendResult(ok=False, response={"error": "rate_limited_429"})
