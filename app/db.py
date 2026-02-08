from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timezone
import logging

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING
from pymongo.errors import OperationFailure

from .settings import settings

log = logging.getLogger("ktzh")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _keys_list(keys: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    return [(k, int(v)) for k, v in keys]


class MongoStore:
    def __init__(self) -> None:
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.sessions = None
        self.messages = None
        self.cases = None
        self.enabled: bool = False

    async def _ensure_index(self, coll, keys: List[Tuple[str, int]], **opts) -> None:
        """
        Создаёт индекс только если такого key-pattern ещё нет.
        Не падает, если Mongo ругается на конфликт имени/опций (code 85).
        """
        keys_norm = _keys_list(keys)

        # 1) если индекс уже есть с таким key pattern — ничего не делаем
        try:
            async for idx in coll.list_indexes():
                existing = list(idx.get("key", {}).items())
                existing = _keys_list(existing)
                if existing == keys_norm:
                    # если хотели unique, а он не unique — логнем
                    if opts.get("unique") and not idx.get("unique", False):
                        log.warning("Mongo index exists but NOT unique for %s on %s", keys_norm, coll.name)
                    return
        except Exception as e:
            # если list_indexes нельзя/ошибка — просто попробуем create_index и обработаем конфликт
            log.warning("Mongo list_indexes failed for %s: %s", getattr(coll, "name", "unknown"), e)

        # 2) пробуем создать
        try:
            await coll.create_index(keys, **opts)
        except OperationFailure as e:
            if getattr(e, "code", None) == 85:
                # IndexOptionsConflict / different name — не валим сервис
                log.warning("Mongo index conflict (code 85) for %s on %s: %s", keys_norm, coll.name, e)
                return
            raise

    async def connect(self) -> None:
        uri = (settings.MONGODB_URI or "").strip()
        if not uri:
            self.enabled = False
            return

        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[settings.DB_NAME]

        self.sessions = self.db[settings.COL_SESSIONS]
        self.messages = self.db[settings.COL_MESSAGES]
        self.cases = self.db[settings.COL_CASES]

        await self.db.command("ping")

        # ✅ индексы (без падения)
        await self._ensure_index(self.sessions, [("chatIdHash", ASCENDING)], unique=True)
        await self._ensure_index(self.messages, [("chatIdHash", ASCENDING), ("createdAt", ASCENDING)])
        await self._ensure_index(self.cases, [("chatIdHash", ASCENDING), ("status", ASCENDING), ("type", ASCENDING)])
        # если у тебя уже есть unique caseId_1 — будет ок, мы просто будем писать caseId в документе

        self.enabled = True

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

        created = doc.get("createdAt") or utcnow().isoformat()
        doc["chatIdHash"] = chat_id_hash
        doc["updatedAt"] = utcnow().isoformat()

        # 🔥 важно: не пишем createdAt в $set, иначе конфликт с $setOnInsert
        doc.pop("_id", None)
        doc.pop("createdAt", None)

        await self.sessions.update_one(
            {"chatIdHash": chat_id_hash},
            {"$set": doc, "$setOnInsert": {"createdAt": created}},
            upsert=True,
        )

    async def add_message(self, doc: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        d = dict(doc)
        d.pop("_id", None)
        d.setdefault("createdAt", utcnow().isoformat())
        await self.messages.insert_one(d)

    async def create_case(self, doc: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        d = dict(doc)
        d.pop("_id", None)
        d.setdefault("createdAt", utcnow().isoformat())
        d.setdefault("updatedAt", utcnow().isoformat())
        await self.cases.insert_one(d)
