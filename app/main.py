from __future__ import annotations

from typing import Any, Dict, Optional, List
import hashlib
import logging

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .settings import settings
from .db import MongoStore
from .dialog import DialogManager
from .wazzup_client import WazzupClient
from .ops_api import router as ops_router

app.include_router(ops_router)

log = logging.getLogger("ktzh")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.APP_NAME)

store = MongoStore()
dialog = DialogManager(store)
wazzup = WazzupClient(settings.WAZZUP_API_KEY)


@app.on_event("startup")
async def startup():
    await store.connect()
    if getattr(store, "enabled", False):
        log.info("Mongo: ENABLED ✅")
    else:
        log.warning("Mongo: DISABLED ⚠️ (set MONGODB_URI / MONGO_URI in Render ENV)")


@app.on_event("shutdown")
async def shutdown():
    await store.close()
    if hasattr(wazzup, "close"):
        await wazzup.close()


def chat_hash(chat_id: str) -> str:
    raw = (settings.PHONE_HASH_SALT + "|" + (chat_id or "")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _check_token(request: Request) -> None:
    if settings.WEBHOOK_TOKEN:
        token = request.query_params.get("token") or request.query_params.get("crmKey")
        if token != settings.WEBHOOK_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid webhook token")


def _payload_to_items(payload: Any) -> List[Dict[str, Any]]:
    """
    Wazzup может прислать:
    - {"messages":[...]}
    - {"statuses":[...]}
    - single message dict
    - list of dicts / list of wrappers
    Возвращаем ТОЛЬКО message dict'ы.
    """
    items: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        for x in payload:
            items.extend(_payload_to_items(x))
        return items

    if not isinstance(payload, dict):
        return []

    # wrappers
    if isinstance(payload.get("messages"), list):
        return [m for m in payload["messages"] if isinstance(m, dict)]

    if isinstance(payload.get("statuses"), list) or "statuses" in payload:
        return []

    # single message dict
    return [payload]


def extract_inbound(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Принимаем ТОЛЬКО один message dict (НЕ wrapper).
    """
    # statuses callbacks / wrapper сюда не должны попадать
    if "statuses" in payload or "messages" in payload:
        return None

    # echo from our bot
    if payload.get("isEcho") is True:
        return None

    # if direction exists and not inbound
    direction = payload.get("direction")
    if direction and direction != "inbound":
        return None

    # ✅ Wazzup часто кладёт text прямо в корень
    text = payload.get("text")

    # fallback legacy formats
    if not text:
        raw = payload.get("raw") or {}
        if isinstance(raw, dict):
            text = raw.get("text")

    if not text:
        content = payload.get("content") or {}
        if isinstance(content, dict):
            text = content.get("text")

    if not text:
        return None

    return {
        "chatId": str(payload.get("chatId") or ""),
        "channelId": str(payload.get("channelId") or ""),
        "chatType": str(payload.get("chatType") or "whatsapp"),
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


async def process_items(items: List[Dict[str, Any]]) -> None:
    log.info("WEBHOOK: got %s item(s)", len(items))

    for item in items:
        msg = extract_inbound(item)

        if not msg or not msg["chatId"]:
            log.info("SKIP: not inbound text payload keys=%s", list(item.keys())[:20])
            continue

        # ✅ TEST MODE: обрабатываем/отвечаем только на один chatId (+ optional channelId)
        if getattr(settings, "TEST_MODE", False):
            allowed_chat = (getattr(settings, "TEST_CHAT_ID", "") or "").strip()
            allowed_channel = (getattr(settings, "TEST_CHANNEL_ID", "") or "").strip()

            if not allowed_chat:
                log.warning("TEST_MODE enabled but TEST_CHAT_ID is empty -> skipping all")
                continue

            if msg["chatId"] != allowed_chat:
                log.info("TEST_MODE: SKIP chatId=%s (allowed=%s)", msg["chatId"], allowed_chat)
                continue

            if allowed_channel and msg["channelId"] != allowed_channel:
                log.info("TEST_MODE: SKIP channelId=%s (allowed=%s)", msg["channelId"], allowed_channel)
                continue

        chat_id_hash = chat_hash(msg["chatId"])
        log.info("IN: chatId=%s text=%r", msg["chatId"], msg["text"])

        # ✅ пишем входящее в Mongo (если включен store)
        await store.add_message({
            "dir": "in",
            "chatIdHash": chat_id_hash,
            "chatId": msg["chatId"],
            "channelId": msg["channelId"],
            "chatType": msg["chatType"],
            "text": msg["text"],
            "raw": msg["raw"],
        })

        try:
            bot_reply = await dialog.handle(
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
            log.exception("Dialog error: %s", e)
            bot_reply = type("BR", (), {"text": "Извините, произошла ошибка. Попробуйте ещё раз."})()

        log.info("BOT: reply=%r", bot_reply.text)

        if settings.BOT_SEND_ENABLED:
            send_res = await wazzup.send_message(
                chat_id=msg["chatId"],
                channel_id=msg["channelId"],
                chat_type=msg["chatType"],
                text=bot_reply.text,
            )
            log.info("SENT: %s", send_res)

            # ✅ пишем исходящее
            await store.add_message({
                "dir": "out",
                "chatIdHash": chat_id_hash,
                "chatId": msg["chatId"],
                "channelId": msg["channelId"],
                "chatType": msg["chatType"],
                "text": bot_reply.text,
                "send": send_res,
            })


@app.post("/webhooks")
async def webhooks_alias(request: Request, background: BackgroundTasks):
    return await wazzup_webhook(request, background)


@app.post("/webhook/wazzup")
async def wazzup_webhook(request: Request, background: BackgroundTasks):
    _check_token(request)

    payload = await request.json()
    items = _payload_to_items(payload)

    # если пришли только statuses / wrappers — просто 200 OK
    if not items:
        return JSONResponse({"ok": True, "ignored": True})

    background.add_task(process_items, items)
    return JSONResponse({"ok": True, "queued": len(items)})
