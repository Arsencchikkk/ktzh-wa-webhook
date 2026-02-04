import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "whatsapp_mess")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "message")

client: AsyncIOMotorClient | None = None


def get_collection():
    if client is None:
        raise RuntimeError("Mongo client is not initialized")
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


async def connect_mongo():
    global client
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set in .env")
    client = AsyncIOMotorClient(MONGODB_URI)
    col = get_collection()

    # Индексы
    await col.create_index("messageId", unique=True, sparse=True)
    await col.create_index("channelId")
    await col.create_index("chatId")
    await col.create_index("chatType")
    await col.create_index("dateTime")
