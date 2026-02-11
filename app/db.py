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
        keys_norm = _keys_list(keys)

        # 1) если индекс уже есть по key-pattern — ничего не делаем
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

        # 2) создаём, но не падаем на конфликтах/дублях
        try:
            await coll.create_index(keys, **opts)
        except DuplicateKeyError as e:
            log.warning("Mongo index create skipped (duplicate key) for %s on %s: %s", keys_norm, coll.name, e)
            return
        except OperationFailure as e:
            code = getattr(e, "code", None)
            # 85 = IndexOptionsConflict, 11000 = duplicate key
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

        # optional outbox
        col_outbox = getattr(settings, "COL_OPS_OUTBOX", None)
        if col_outbox:
            self.ops_outbox = self.db[col_outbox]

        await self.db.command("ping")

        # ---- индексы sessions/messages/cases ----
        await self._ensure_index(self.sessions, [("chatIdHash", ASCENDING)], unique=True)
        await self._ensure_index(self.messages, [("chatIdHash", ASCENDING), ("createdAt", ASCENDING)])
        await self._ensure_index(self.cases, [("chatIdHash", ASCENDING), ("status", ASCENDING), ("type", ASCENDING)])
        await self._ensure_index(
            self.cases,
            [("chatIdHash", ASCENDING), ("status", ASCENDING), ("updatedAt", DESCENDING)],
        )
        await self._ensure_index(self.cases, [("caseId", ASCENDING)], unique=True)

        # ---- индексы outbox (если включён) ----
        if self.ops_outbox is not None:
            await self._ensure_index(self.ops_outbox, [("status", ASCENDING), ("nextAttemptAt", ASCENDING)])
            await self._ensure_index(self.ops_outbox, [("lockUntil", ASCENDING)])
            await self._ensure_index(self.ops_outbox, [("kind", ASCENDING), ("caseId", ASCENDING)], unique=True)

        self.enabled = True

    async def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.enabled = False

    # ---------- sessions ----------
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

        # createdAt не должен быть в $set
        doc.pop("_id", None)
        doc.pop("createdAt", None)

        await self.sessions.update_one(
            {"chatIdHash": chat_id_hash},
            {"$set": doc, "$setOnInsert": {"createdAt": created}},
            upsert=True,
        )

    # ---------- messages ----------
    async def add_message(self, doc: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        d = dict(doc)
        d.pop("_id", None)
        d.setdefault("createdAt", utcnow().isoformat())
        await self.messages.insert_one(d)

    # ---------- cases ----------
    async def create_case(self, doc: Dict[str, Any]) -> None:
        """
        Идемпотентное создание кейса.
        updatedAt обновляем ТОЛЬКО через $set.
        """
        if not self.enabled:
            return

        d = dict(doc)
        d.pop("_id", None)

        # гарантируем caseId
        if not d.get("caseId"):
            if d.get("ticketId"):
                d["caseId"] = str(d["ticketId"])
            else:
                d["caseId"] = f"KTZH-{utcnow().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3).upper()}"

        # гарантируем payload.followups
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

    # ---------- outbox ----------
    async def enqueue_ops_outbox(
        self,
        *,
        kind: str,
        case_id: str,
        case_type: str,
        text: str,
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> None:
        if not self.enabled or self.ops_outbox is None:
            return

        now = utcnow().isoformat()

        doc_set = {
            "caseType": case_type,
            "text": text,
            "source": source,
            "target": target,
            "status": "pending",
            "updatedAt": now,
            "nextAttemptAt": now,
            "lockUntil": None,
        }

        doc_insert = {
            "kind": kind,
            "caseId": case_id,
            "attempts": 0,
            "createdAt": now,
        }

        await self.ops_outbox.update_one(
            {"kind": kind, "caseId": case_id},
            {"$set": doc_set, "$setOnInsert": doc_insert},
            upsert=True,
        )

    async def claim_pending_outbox(self) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.ops_outbox is None:
            return None

        now_dt = utcnow()
        now = now_dt.isoformat()

        lock_seconds = int(getattr(settings, "OPS_LOCK_SECONDS", 60) or 60)
        max_attempts = int(getattr(settings, "OPS_MAX_ATTEMPTS", 10) or 10)

        lock_until = (now_dt + timedelta(seconds=lock_seconds)).isoformat()

        q = {
            "status": "pending",
            "nextAttemptAt": {"$lte": now},
            "attempts": {"$lt": max_attempts},
            "$or": [{"lockUntil": None}, {"lockUntil": {"$lte": now}}],
        }

        upd = {"$set": {"status": "sending", "lockUntil": lock_until, "updatedAt": now}}

        doc = await self.ops_outbox.find_one_and_update(
            q,
            upd,
            sort=[("nextAttemptAt", ASCENDING), ("createdAt", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )
        return doc

    async def mark_outbox_sent(self, outbox_id, resp: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self.ops_outbox is None:
            return
        now = utcnow().isoformat()
        await self.ops_outbox.update_one(
            {"_id": outbox_id},
            {"$set": {"status": "sent", "sentAt": now, "updatedAt": now, "lockUntil": None, "lastResponse": resp or None}},
        )

    async def mark_outbox_failed(self, outbox_id, error: str, attempts: int) -> None:
        if not self.enabled or self.ops_outbox is None:
            return

        base = int(getattr(settings, "OPS_BACKOFF_BASE_SECONDS", 10) or 10)
        cap = int(getattr(settings, "OPS_BACKOFF_MAX_SECONDS", 600) or 600)
        max_attempts = int(getattr(settings, "OPS_MAX_ATTEMPTS", 10) or 10)

        delay = min(cap, base * (2 ** max(0, attempts - 1)))

        now_dt = utcnow()
        now = now_dt.isoformat()
        next_at = (now_dt + timedelta(seconds=delay)).isoformat()

        if attempts >= max_attempts:
            await self.ops_outbox.update_one(
                {"_id": outbox_id},
                {"$set": {"status": "failed", "failedAt": now, "updatedAt": now, "lockUntil": None, "lastError": error}},
            )
            return

        await self.ops_outbox.update_one(
            {"_id": outbox_id},
            {
                "$set": {"status": "pending", "updatedAt": now, "lockUntil": None, "lastError": error, "nextAttemptAt": next_at},
                "$inc": {"attempts": 1},
            },
        )
