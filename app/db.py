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

    async def connect(self) -> None:
        if not settings.MONGODB_URI:
            raise RuntimeError("MONGO_URI is empty. Set it in Render ENV.")

        self.client = AsyncIOMotorClient(settings.MONGODB_URI)
        self.db = self.client[settings.DB_NAME]

        self.sessions = self.db[settings.COL_SESSIONS]
        self.messages = self.db[settings.COL_MESSAGES]
        self.cases = self.db[settings.COL_CASES]

        # ping
        await self.db.command("ping")

        # indexes
        await self.sessions.create_index([("chatIdHash", ASCENDING)], unique=True)
        await self.messages.create_index([("chatIdHash", ASCENDING), ("createdAt", ASCENDING)])
        await self.cases.create_index([("chatIdHash", ASCENDING), ("status", ASCENDING), ("type", ASCENDING)])

    async def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    async def get_session(self, chat_id_hash: str) -> Optional[Dict[str, Any]]:
        doc = await self.sessions.find_one({"chatIdHash": chat_id_hash})
        return doc

    async def save_session(self, chat_id_hash: str, session: Dict[str, Any]) -> None:
        session = dict(session)
        session["chatIdHash"] = chat_id_hash
        session["updatedAt"] = utcnow().isoformat()
        session.setdefault("createdAt", utcnow().isoformat())

        await self.sessions.update_one(
            {"chatIdHash": chat_id_hash},
            {"$set": session, "$setOnInsert": {"createdAt": session["createdAt"]}},
            upsert=True,
        )

    async def add_message(self, doc: Dict[str, Any]) -> None:
        doc = dict(doc)
        doc.setdefault("createdAt", utcnow().isoformat())
        await self.messages.insert_one(doc)

    async def create_case(self, doc: Dict[str, Any]) -> None:
        doc = dict(doc)
        doc.setdefault("createdAt", utcnow().isoformat())
        doc.setdefault("updatedAt", utcnow().isoformat())
        await self.cases.insert_one(doc)
