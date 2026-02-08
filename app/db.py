from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING

from .settings import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MongoStore:
    def __init__(self) -> None:
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.sessions = None
        self.messages = None
        self.cases = None
        self.enabled: bool = False  # ✅ для логов

    async def connect(self) -> None:
        uri = (settings.MONGODB_URI or "").strip()
        if not uri:
            # хочешь — можно raise, чтобы сервис падал и ты точно видел проблему
            self.enabled = False
            return

        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[settings.DB_NAME]

        self.sessions = self.db[settings.COL_SESSIONS]
        self.messages = self.db[settings.COL_MESSAGES]
        self.cases = self.db[settings.COL_CASES]

        await self.db.command("ping")

        await self.sessions.create_index([("chatIdHash", ASCENDING)], unique=True)
        await self.messages.create_index([("chatIdHash", ASCENDING), ("createdAt", ASCENDING)])
        await self.cases.create_index([("chatIdHash", ASCENDING), ("status", ASCENDING), ("type", ASCENDING)])

        self.enabled = True

    async def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    async def get_session(self, chat_id_hash: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        return await self.sessions.find_one({"chatIdHash": chat_id_hash})

    async def save_session(self, chat_id_hash: str, session: Dict[str, Any]) -> None:
        doc = dict(session)

    created = doc.get("createdAt") or utcnow().isoformat()
    doc["chatIdHash"] = chat_id_hash
    doc["updatedAt"] = utcnow().isoformat()

    # 🔥 важно: createdAt НЕ должен попасть в $set, иначе конфликт с $setOnInsert
    d None)

    await self.sessions.update_one(
        {"chatIdHash": chat_id_hash},
        {
            "$set": doc,
            "$setOnInsert": {"createdAt": created},
        },
        upsert=True,
    )



    async def add_message(self, doc: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        doc = dict(doc)
        doc.setdefault("createdAt", utcnow().isoformat())
        await self.messages.insert_one(doc)

    async def create_case(self, doc: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        doc = dict(doc)
        doc.setdefault("createdAt", utcnow().isoformat())
        doc.setdefault("updatedAt", utcnow().isoformat())
        await self.cases.insert_one(doc)
