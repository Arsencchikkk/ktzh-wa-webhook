from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timezone
import logging
import secrets

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import OperationFailure, DuplicateKeyError

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
        Не падает:
          - code 85: IndexOptionsConflict
          - code 11000: duplicate key (например при unique индексе на старых данных)
        """
        keys_norm = _keys_list(keys)

        # 1) если индекс уже есть с таким key pattern — ничего не делаем
        try:
            async for idx in coll.list_indexes():
                existing = list(idx.get("key", {}).items())
                existing = _keys_list(existing)
                if existing == keys_norm:
                    if opts.get("unique") and not idx.get("unique", False):
                        log.warning("Mongo index exists but NOT unique for %s on %s", keys_norm, coll.name)
                    return
        except Exception as e:
            log.warning("Mongo list_indexes failed for %s: %s", getattr(coll, "name", "unknown"), e)

        # 2) пробуем создать
        try:
            await coll.create_index(keys, **opts)
        except OperationFailure as e:
            code = getattr(e, "code", None)
            if code in (85, 11000):
                log.warning("Mongo index create skipped (code %s) for %s on %s: %s", code, keys_norm, coll.name, e)
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

        # ✅ быстрый поиск последней open-заявки
        await self._ensure_index(self.cases, [("chatIdHash", ASCENDING), ("status", ASCENDING), ("updatedAt", DESCENDING)])

        # ✅ уникальный caseId (не падаем при старых дублях)
        await self._ensure_index(self.cases, [("caseId", ASCENDING)], unique=True)

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

        # ✅ гарантируем caseId
        if not d.get("caseId"):
            if d.get("ticketId"):
                d["caseId"] = str(d["ticketId"])
            else:
                d["caseId"] = f"KTZH-{utcnow().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3).upper()}"

        # ✅ гарантируем payload.followups
        payload = d.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("followups", [])
        d["payload"] = payload

        d.setdefault("createdAt", utcnow().isoformat())
        d.setdefault("updatedAt", utcnow().isoformat())

        # ✅ идемпотентно: если вдруг повторно пришёл create с тем же caseId — не падаем
        try:
            await self.cases.update_one(
                {"caseId": d["caseId"]},
                {"$setOnInsert": d, "$set": {"updatedAt": utcnow().isoformat()}},
                upsert=True,
            )
        except DuplicateKeyError as e:
            log.warning("create_case: duplicate key for caseId=%s: %s", d.get("caseId"), e)

    async def get_last_open_case(self, chat_id_hash: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        return await self.cases.find_one(
            {"chatIdHash": chat_id_hash, "status": "open"},
            sort=[("updatedAt", DESCENDING), ("createdAt", DESCENDING)],
        )

    async def append_case_followup(self, case_id: str, note: Dict[str, Any]) -> bool:
        """
        Добавляет дополнение в payload.followups[] и обновляет updatedAt.
        Пишем ТОЛЬКО в open кейс.
        """
        if not self.enabled:
            return False

        n = dict(note or {})
        n.setdefault("ts", utcnow().isoformat())
        n.setdefault("text", "")

        res = await self.cases.update_one(
            {"caseId": case_id, "status": "open"},
            {"$push": {"payload.followups": n}, "$set": {"updatedAt": utcnow().isoformat()}},
        )

        if res.matched_count == 0:
            log.warning("append_case_followup: open case not found: %s", case_id)
            return False

        return True
