from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

from . import settings

@dataclass
class Mongo:
    client: AsyncIOMotorClient
    db: AsyncIOMotorDatabase
    messages: AsyncIOMotorCollection
    sessions: AsyncIOMotorCollection
    cases: AsyncIOMotorCollection

mongo: Optional[Mongo] = None

async def init_mongo() -> Mongo:
    global mongo
    if mongo is not None:
        return mongo

    if not settings.MONGODB_URI:
        raise RuntimeError("MONGODB_URI is empty")

    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client[settings.DB_NAME]

    messages = db[settings.COL_MESSAGES]
    sessions = db[settings.COL_SESSIONS]
    cases = db[settings.COL_CASES]

    mongo = Mongo(client=client, db=db, messages=messages, sessions=sessions, cases=cases)
    await ensure_indexes(mongo)
    return mongo

async def close_mongo() -> None:
    global mongo
    if mongo and mongo.client:
        mongo.client.close()
    mongo = None

async def ensure_indexes(m: Mongo) -> None:
    idx_info = await m.messages.index_information()

    # 1) drop legacy unique indexes that break inserts (null duplicates)
    for name, spec in idx_info.items():
        keys = spec.get("key") or []
        key_fields = [k for k, _ in keys]

        if ("wa_message_id" in key_fields) or ("waMessageId" in key_fields):
            await m.messages.drop_index(name)

    # (на всякий) дроп по имени, если вдруг есть
    idx_info = await m.messages.index_information()
    if "wa_message_id_1" in idx_info:
        await m.messages.drop_index("wa_message_id_1")

    # 2) our idempotency key
    await m.messages.create_index("messageId", unique=True, sparse=True, name="messageId_1")

    # useful query indexes
    await m.messages.create_index([("channelId", 1), ("chatId", 1), ("dateTime", 1)], name="chat_time_1")
    await m.messages.create_index("direction", name="direction_1")
    await m.messages.create_index("currentStatus", name="currentStatus_1")

    # Sessions unique per chat
    await m.sessions.create_index([("channelId", 1), ("chatId", 1)], unique=True, name="session_chat_1")
    await m.sessions.create_index("updatedAt", name="session_updatedAt_1")

    # Cases
    await m.cases.create_index("caseId", unique=True, name="caseId_1")
    await m.cases.create_index("status", name="case_status_1")
    await m.cases.create_index([("channelId", 1), ("chatId", 1), ("createdAt", -1)], name="case_chat_createdAt_1")
