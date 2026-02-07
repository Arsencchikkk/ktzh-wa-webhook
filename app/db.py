from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from pymongo import MongoClient, ReturnDocument
from pymongo.collection import Collection

from settings import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MongoStore:
    def __init__(self) -> None:
        if not settings.MONGODB_URI:
            raise RuntimeError("MONGODB_URI is empty. Set it in ENV for Render/local.")
        self.client = MongoClient(settings.MONGODB_URI)
        self.db = self.client[settings.DB_NAME]
        self.messages: Collection = self.db[settings.COL_MESSAGES]
        self.sessions: Collection = self.db[settings.COL_SESSIONS]
        self.cases: Collection = self.db[settings.COL_CASES]

        # indexes (safe to call repeatedly)
        self.sessions.create_index("chatIdHash", unique=True)
        self.cases.create_index([("chatIdHash", 1), ("status", 1), ("type", 1)])
        self.messages.create_index([("chatIdHash", 1), ("dateTime", 1)])

    # ---------- sessions ----------
    def get_session(self, chat_id_hash: str) -> Optional[Dict[str, Any]]:
        return self.sessions.find_one({"chatIdHash": chat_id_hash})

    def upsert_session(self, chat_id_hash: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        patch["updatedAt"] = utcnow()
        return self.sessions.find_one_and_update(
            {"chatIdHash": chat_id_hash},
            {"$set": patch, "$setOnInsert": {"createdAt": utcnow(), "chatIdHash": chat_id_hash}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def reset_session(self, chat_id_hash: str) -> None:
        self.sessions.update_one(
            {"chatIdHash": chat_id_hash},
            {"$set": {
                "shared": {"train": None, "carNumber": None},
                "pending": {"slots": [], "bundle": None, "caseTypes": []},
                "cases": {},
                "lastBot": {"text": None, "askedSlots": []},
                "updatedAt": utcnow(),
            }},
            upsert=True,
        )

    # ---------- messages ----------
    def add_message(self, doc: Dict[str, Any]) -> None:
        doc.setdefault("createdAt", utcnow())
        self.messages.insert_one(doc)

    # ---------- cases ----------
    def create_case(self, chat_id_hash: str, case_type: str, payload: Dict[str, Any]) -> str:
        """
        Creates a local case record (not an external CRM ticket).
        Returns generated ticketId.
        """
        ticket_id = payload.get("ticketId")
        if not ticket_id:
            ticket_id = self._gen_ticket_id(case_type)

        doc = {
            "ticketId": ticket_id,
            "chatIdHash": chat_id_hash,
            "type": case_type,
            "status": "open",
            "payload": payload,
            "createdAt": utcnow(),
            "updatedAt": utcnow(),
        }
        self.cases.insert_one(doc)
        return ticket_id

    def close_case(self, chat_id_hash: str, ticket_id: str) -> None:
        self.cases.update_one(
            {"chatIdHash": chat_id_hash, "ticketId": ticket_id},
            {"$set": {"status": "closed", "updatedAt": utcnow()}}
        )

    def _gen_ticket_id(self, case_type: str) -> str:
        # Example: KTZH-20260207-LOST-8F21C2A1
        import secrets
        from datetime import datetime
        dt = datetime.utcnow().strftime("%Y%m%d")
        suffix = secrets.token_hex(4).upper()
        prefix = "LOST" if case_type == "lost_and_found" else ("THANKS" if case_type == "gratitude" else "CMP")
        return f"KTZH-{dt}-{prefix}-{suffix}"
