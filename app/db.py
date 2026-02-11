from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
import logging
import secrets

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, ReturnDocument
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
        self.ops_outbox = None
        self.enabled: bool = False

    async def _ensure_index(self, coll, keys: List[Tuple[str, int]], **opts) -> None:
        """
        Создаёт индекс только если такого key-pattern ещё нет.
        Не падает:
          - code 85: IndexOptionsConflict
          - duplicate data при unique индексе (OperationFailure/DuplicateKeyError)
        """
        keys_norm = _keys_list(keys)

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

        try:
            await coll.create_index(keys, **opts)
        except DuplicateKeyError as e:
            log.warning("Mongo index create skipped (duplicate key) for %s on %s: %s", keys_norm, coll.name, e)
            return
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
        self.ops_outbox = self.db[getattr(settings, "COL_OPS_OUTBOX", "ops_outbox")]

        await self.db.command("ping")

        # ✅ индексы
        await self._ensure_index(self.sessions, [("chatIdHash", ASCENDING)], unique=True)
        await self._ensure_index(self.messages, [("chatIdHash", ASCENDING), ("createdAt", ASCENDING)])
        await self._ensure_index(self.cases, [("chatIdHash", ASCENDING), ("status", ASCENDING), ("type", ASCENDING)])
        await self._ensure_index(self.cases, [("chatIdHash", ASCENDING), ("status", ASCENDING), ("updatedAt", DESCENDING)])
        await self._ensure_index(self.cases, [("caseId", ASCENDING)], unique=True)

        # ✅ outbox индексы (воркер)
        await self._ensure_index(self.ops_outbox, [("status", ASCENDING), ("nextAttemptAt", ASCENDING), ("createdAt", DESCENDING)])
        await self._ensure_index(self.ops_outbox, [("lockUntil", ASCENDING)])
        await self._ensure_index(self.ops_outbox, [("caseId", ASCENDING), ("kind", ASCENDING)])

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
        """
        Идемпотентное создание кейса.
        КРИТИЧНО: updatedAt обновляем ТОЛЬКО через $set, чтобы не было code 40 conflict.
        """
        if not self.enabled:
            return

        d = dict(doc)
        d.pop("_id", None)

        if not d.get("caseId"):
            if d.get("ticketId"):
                d["caseId"] = str(d["ticketId"])
            else:
                d["caseId"] = f"KTZH-{utcnow().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3).upper()}"

        payload = d.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("followups", [])
        d["payload"] = payload

        now = utcnow().isoformat()
        created = d.get("createdAt") or now

        d.pop("createdAt", None)
        d.pop("updatedAt", None)

        insert_doc = dict(d)
        insert_doc["createdAt"] = created

        try:
            await self.cases.update_one(
                {"caseId": d["caseId"]},
                {
                    "$setOnInsert": insert_doc,
                    "$set": {"updatedAt": now},
                },
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

    # ===================== OPS OUTBOX =====================

    async def enqueue_ops_message(self, payload: Dict[str, Any]) -> None:
        """Кладём сообщение в ops_outbox со статусом pending."""
        if not self.enabled or self.ops_outbox is None:
            return

        now = utcnow().isoformat()
        d = dict(payload)
        d.pop("_id", None)
        d.setdefault("status", "pending")          # pending|processing|sent|failed
        d.setdefault("attempts", 0)
        d.setdefault("createdAt", now)
        d.setdefault("updatedAt", now)
        d.setdefault("nextAttemptAt", now)        # можно отложить
        d.setdefault("lockUntil", None)

        await self.ops_outbox.insert_one(d)

    async def claim_next_ops_message(self, lock_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Атомарно "забираем" 1 pending сообщение, ставим processing + lockUntil.
        Чтобы несколько воркеров не отправляли одно и то же.
        """
        if not self.enabled or self.ops_outbox is None:
            return None

        now_dt = utcnow()
        now = now_dt.isoformat()
        lock_until = (now_dt + timedelta(seconds=lock_seconds)).isoformat()

        doc = await self.ops_outbox.find_one_and_update(
            filter={
                "status": "pending",
                "nextAttemptAt": {"$lte": now},
                "$or": [{"lockUntil": None}, {"lockUntil": {"$lte": now}}],
            },
            update={
                "$set": {"status": "processing", "lockUntil": lock_until, "updatedAt": now},
                "$inc": {"attempts": 1},
            },
            sort=[("nextAttemptAt", ASCENDING), ("createdAt", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )
        return doc

    async def mark_ops_sent(self, outbox_id, response: Dict[str, Any]) -> None:
        if not self.enabled or self.ops_outbox is None:
            return
        now = utcnow().isoformat()
        await self.ops_outbox.update_one(
            {"_id": outbox_id},
            {"$set": {"status": "sent", "sentAt": now, "updatedAt": now, "lockUntil": None, "response": response}},
        )

    async def mark_ops_failed(self, outbox_id, error: str, retry_after_seconds: int, give_up: bool = False) -> None:
        if not self.enabled or self.ops_outbox is None:
            return
        now_dt = utcnow()
        now = now_dt.isoformat()
        next_at = (now_dt + timedelta(seconds=retry_after_seconds)).isoformat()

        if give_up:
            await self.ops_outbox.update_one(
                {"_id": outbox_id},
                {"$set": {"status": "failed", "updatedAt": now, "lockUntil": None, "error": error}},
            )
            return

        await self.ops_outbox.update_one(
            {"_id": outbox_id},
            {"$set": {"status": "pending", "updatedAt": now, "lockUntil": None, "error": error, "nextAttemptAt": next_at}},
        )
