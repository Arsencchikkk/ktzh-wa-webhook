from __future__ import annotations

from typing import Any, Dict, Optional
import hashlib
import logging

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .settings import settings
from .db import MongoStore
from .dialog import DialogManager
from .wazzup_client import WazzupClient

log = logging.getLogger("ktzh-bot")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.APP_NAME)

store = MongoStore()
dialog = DialogManager(store)
wazzup = WazzupClient(settings.WAZZUP_API_KEY)


@app.on_event("startup")
async def startup():
    await store.connect()


@app.on_event("shutdown")
async def shutdown():
    await store.close()
    await wazzup.close()


def chat_hash(chat_id: str) -> str:
    raw = (settings.PHONE_HASH_SALT + "|" + (chat_id or "")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def extract_inbound(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # statuses callbacks
    if "statuses" in payload:
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


def _check_token(request: Request) -> None:
    if settings.WEBHOOK_TOKEN:
        token = request.query_params.get("token") or request.query_params.get("crmKey")
        if token != settings.WEBHOOK_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid webhook token")


@app.get("/")
def root():
    return {"ok": True, "service": settings.APP_NAME}


@app.get("/health")
def health():
    return {"ok": True}


async def process_items(items: list) -> None:
    log.info("WEBHOOK: got %s item(s)", len(items))

    for item in items:
        msg = extract_inbound(item)
        if not msg or not msg["chatId"]:
            log.info("SKIP: not inbound text payload keys=%s", list(item.keys())[:20])
            continue

        chat_id_hash = chat_hash(msg["chatId"])
        log.info("IN: chatId=%s text=%r", msg["chatId"], msg["text"])

        # ✅ пишем входящее в Mongo
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

            # ✅ пишем исходящее в Mongo
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
    items = payload if isinstance(payload, list) else [payload]

    # ✅ async background ok
    background.add_task(process_items, items)
    return JSONResponse({"ok": True, "queued": len(items)})
