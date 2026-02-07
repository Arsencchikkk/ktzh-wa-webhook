from __future__ import annotations

from typing import Any, Dict, Optional, List
import hashlib
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from .settings import settings
from .db import MongoStore
from .dialog import DialogManager
from .wazzup_client import WazzupClient

logger = logging.getLogger("ktzh-bot")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.APP_NAME)

store = MongoStore()
dialog = DialogManager(store)
wazzup = WazzupClient(settings.WAZZUP_API_KEY)


@app.on_event("startup")
async def startup():
    # если у тебя pymongo — connect/close можно сделать no-op в MongoStore
    if hasattr(store, "connect"):
        res = store.connect()
        if callable(res):
            maybe = res()
            if hasattr(maybe, "__await__"):
                await maybe


@app.on_event("shutdown")
async def shutdown():
    try:
        await wazzup.close()
    except Exception:
        pass

    if hasattr(store, "close"):
        res = store.close()
        if callable(res):
            maybe = res()
            if hasattr(maybe, "__await__"):
                await maybe


def chat_hash(chat_id: str) -> str:
    raw = (settings.PHONE_HASH_SALT + "|" + (chat_id or "")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _check_token(request: Request) -> None:
    if settings.WEBHOOK_TOKEN:
        token = request.query_params.get("token") or request.query_params.get("crmKey")
        if token != settings.WEBHOOK_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid webhook token")


def normalize_items(payload: Any) -> List[Dict[str, Any]]:
    """
    Wazzup может прислать:
    - list сообщений
    - dict с ключами messages/items/data/events
    - один message dict
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("messages", "items", "data", "events"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [payload]

    return []


def extract_inbound(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # фильтры inbound/echo
    direction = payload.get("direction")
    is_echo = payload.get("isEcho")
    if direction and direction != "inbound":
        return None
    if is_echo is True:
        return None

    # текст может быть в разных местах
    raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
    text = raw.get("text")

    if not text and isinstance(payload.get("content"), dict):
        text = payload["content"].get("text")

    if not text and isinstance(payload.get("message"), dict):
        text = payload["message"].get("text")

    if not text:
        text = payload.get("text")

    if not text:
        return None

    chat_id = payload.get("chatId") or (payload.get("chat") or {}).get("id")
    channel_id = payload.get("channelId") or (payload.get("channel") or {}).get("id")
    chat_type = payload.get("chatType") or payload.get("type") or "whatsapp"

    if not chat_id:
        return None

    return {
        "chatId": str(chat_id),
        "channelId": str(channel_id or ""),
        "chatType": str(chat_type),
        "dateTime": payload.get("dateTime"),
        "text": str(text),
        "raw": payload,
    }


@app.get("/")
def root():
    return {"ok": True, "service": settings.APP_NAME}


@app.get("/health")
def health():
    return {"ok": True}


async def process_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    handled = 0
    results = []

    for item in items:
        msg = extract_inbound(item)
        if not msg:
            logger.info("SKIP: not inbound text payload keys=%s", list(item.keys())[:20])
            continue

        chat_id = msg["chatId"]
        chat_id_hash = chat_hash(chat_id)

        logger.info("IN: chatId=%s text=%r", chat_id, msg["text"])

        try:
            bot_reply = dialog.handle(
    chat_id_hash=chat_id_hash,
    chat_meta={
        "chatId": msg["chatId"],
        "channelId": msg["channelId"],
        "chatType": msg["chatType"],
        "dateTime": msg["dateTime"],
        "raw": msg["raw"],
    },
    user_text=msg["text"],
)

        except Exception as e:
            logger.exception("Dialog error: %s", e)
            continue

        logger.info("BOT: reply=%r", bot_reply.text)

        send_res = {"ok": True, "skipped": True}
        if settings.BOT_SEND_ENABLED:
            send_res = await wazzup.send_message(
                chat_id=msg["chatId"],
                channel_id=msg["channelId"],
                chat_type=msg["chatType"],
                text=bot_reply.text,
            )
            logger.info("SENT: %s", send_res)

        handled += 1
        results.append({"chatId": msg["chatId"], "reply": bot_reply.text, "send": send_res})

    return {"ok": True, "handled": handled, "results": results}


# ✅ Alias для Wazzup
@app.post("/webhooks")
async def webhooks_alias(request: Request):
    return await wazzup_webhook(request)


@app.post("/webhook/wazzup")
async def wazzup_webhook(request: Request):
    _check_token(request)

    payload = await request.json()
    items = normalize_items(payload)

    logger.info("WEBHOOK: got %d item(s)", len(items))

    # ВАЖНО: делаем синхронно (надежно). Если хочешь — потом вернём background.
    res = await process_items(items)
    return JSONResponse(res)
