from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime, timezone
import logging

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING

from .settings import settings

log = logging.getLogger("ktzh")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MongoStore:
    def __init__(self) -> None:
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.sessions = None
        self.messages = None
        self.cases = None
        self.enabled: bool = False

    async def connect(self) -> None:
        uri = (settings.MONGODB_URI or "").strip()
        if not uri:
            self.enabled = False
            log.warning("Mongo: DISABLED ⚠️ (set MONGODB_URI / MONGO_URI in Render ENV)")
            return

        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[settings.DB_NAME]

        self.sessions = self.db[settings.COL_SESSIONS]
        self.messages = self.db[settings.COL_MESSAGES]
        self.cases = self.db[settings.COL_CASES]

        await self.db.command("ping")

        # --- DROP legacy/bad unique indexes that cause dup-key with nulls ---

        # sessions: legacy unique (channelId, chatId)
        async for idx in self.sessions.list_indexes():
            name = idx.get("name")
            key = idx.get("key") or {}
            if name == "session_chat_1" or key == {"channelId": 1, "chatId": 1}:
                await self.sessions.drop_index(name)
                log.warning("Mongo: dropped legacy sessions index %s", name)

        # cases: legacy unique caseId (your inserts had no caseId before)
        async for idx in self.cases.list_indexes():
            name = idx.get("name")
            key = idx.get("key") or {}
            if name == "caseId_1" or key == {"caseId": 1}:
                await self.cases.drop_index(name)
                log.warning("Mongo: dropped legacy cases index %s", name)

        # --- correct indexes ---

        await self.sessions.create_index(
            [("chatIdHash", ASCENDING)],
            unique=True,
            name="session_chatIdHash_1",
        )
        await self.messages.create_index([("chatIdHash", ASCENDING), ("createdAt", ASCENDING)])
        await self.cases.create_index([("chatIdHash", ASCENDING), ("status", ASCENDING), ("type", ASCENDING)])

        # unique ticketId for cases
        await self.cases.create_index([("ticketId", ASCENDING)], unique=True, name="ticketId_1")

        self.enabled = True
        log.info("Mongo: ENABLED ✅ db=%s", settings.DB_NAME)

    async def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.enabled = False

    async def get_session(self, chat_id_hash: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        return await self.sessions.find_one({"chatIdHash": chat_id_hash})

    async def save_session(self, chat_id_hash: str, session: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        doc = dict(session)
        doc.pop("_id", None)

        created = doc.get("createdAt") or utcnow().isoformat()

        # IMPORTANT: createdAt must NOT be in $set if you also use $setOnInsert
        doc.pop("createdAt", None)

        doc["chatIdHash"] = chat_id_hash
        doc["updatedAt"] = utcnow().isoformat()

        await self.sessions.update_one(
            {"chatIdHash": chat_id_hash},
            {"$set": doc, "$setOnInsert": {"createdAt": created}},
            upsert=True,
        )

    async def add_message(self, doc: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        d = dict(doc)
        d.setdefault("createdAt", utcnow().isoformat())
        await self.messages.insert_one(d)

    async def create_case(self, doc: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        d = dict(doc)
        d.setdefault("createdAt", utcnow().isoformat())
        d.setdefault("updatedAt", utcnow().isoformat())

        # safety: if someone forgot
        if "ticketId" not in d and "caseId" in d:
            d["ticketId"] = d["caseId"]
        if "caseId" not in d and "ticketId" in d:
            d["caseId"] = d["ticketId"]

        await self.cases.insert_one(d)
