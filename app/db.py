from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
import logging

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
        self.ops_outbox = self.db[settings.COL_OPS_OUTBOX]

        await self.db.command("ping")

        # sessions/messages/cases индексы (как у тебя)
        await self._ensure_index(self.sessions, [("chatIdHash", ASCENDING)], unique=True)
        await self._ensure_index(self.messages, [("chatIdHash", ASCENDING), ("createdAt", ASCENDING)])
        await self._ensure_index(self.cases, [("chatIdHash", ASCENDING), ("status", ASCENDING), ("type", ASCENDING)])
        await self._ensure_index(self.cases, [("chatIdHash", ASCENDING), ("status", ASCENDING), ("updatedAt", DESCENDING)])
        await self._ensure_index(self.cases, [("caseId", ASCENDING)], unique=True)

        # ✅ outbox индексы
        await self._ensure_index(self.ops_outbox, [("status", ASCENDING), ("nextAttemptAt", ASCENDING)])
        await self._ensure_index(self.ops_outbox, [("lockUntil", ASCENDING)])
        await self._ensure_index(self.ops_outbox, [("kind", ASCENDING), ("caseId", ASCENDING)], unique=True)

        self.enabled = True

    async def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.enabled = False

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
        """
        Идемпотентно кладём в ops_outbox.
        Если запись уже есть — обновим text/target и снова поставим pending.
        """
        if not self.enabled:
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
        """
        Забираем 1 pending задачу с учётом lockUntil/nextAttemptAt.
        """
        if not self.enabled:
            return None

        now_dt = utcnow()
        now = now_dt.isoformat()
        lock_until = (now_dt + timedelta(seconds=int(settings.OPS_LOCK_SECONDS))).isoformat()

        q = {
            "status": "pending",
            "nextAttemptAt": {"$lte": now},
            "attempts": {"$lt": int(settings.OPS_MAX_ATTEMPTS)},
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
        if not self.enabled:
            return
        now = utcnow().isoformat()
        await self.ops_outbox.update_one(
            {"_id": outbox_id},
            {"$set": {"status": "sent", "sentAt": now, "updatedAt": now, "lockUntil": None, "lastResponse": resp or None}},
        )

    async def mark_outbox_failed(self, outbox_id, error: str, attempts: int) -> None:
        if not self.enabled:
            return

        # backoff: base * 2^(attempts-1), capped
        base = int(settings.OPS_BACKOFF_BASE_SECONDS)
        cap = int(settings.OPS_BACKOFF_MAX_SECONDS)
        delay = min(cap, base * (2 ** max(0, attempts - 1)))

        now_dt = utcnow()
        now = now_dt.isoformat()
        next_at = (now_dt + timedelta(seconds=delay)).isoformat()

        if attempts >= int(settings.OPS_MAX_ATTEMPTS):
            await self.ops_outbox.update_one(
                {"_id": outbox_id},
                {"$set": {"status": "failed", "failedAt": now, "updatedAt": now, "lockUntil": None, "lastError": error}},
            )
            return

        await self.ops_outbox.update_one(
            {"_id": outbox_id},
            {
                "$set": {
                    "status": "pending",
                    "updatedAt": now,
                    "lockUntil": None,
                    "lastError": error,
                    "nextAttemptAt": next_at,
                },
                "$inc": {"attempts": 1},
            },
        )
