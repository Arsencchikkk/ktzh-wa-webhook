from __future__ import annotations
from typing import Any, Dict, Optional
import hashlib

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .settings import settings
from .db import MongoStore
from .dialog import DialogManager
from .wazzup_client import WazzupClient

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


def chat_hash(chat_id: str) -> str:
    raw = (settings.PHONE_HASH_SALT + "|" + (chat_id or "")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def extract_inbound(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    direction = payload.get("direction")
    is_echo = payload.get("isEcho")
    if direction and direction != "inbound":
        return None
    if is_echo is True:
        return None

    raw = payload.get("raw") or {}
    text = raw.get("text") if isinstance(raw, dict) else None
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
        "text": text,
        "raw": payload,
    }


@app.get("/")
def root():
    return {"ok": True, "service": settings.APP_NAME}


@app.get("/health")
def health():
    return {"ok": True}


def _check_token(request: Request) -> None:
    if settings.WEBHOOK_TOKEN:
        token = request.query_params.get("token") or request.query_params.get("crmKey")
        if token != settings.WEBHOOK_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid webhook token")


async def _process_items(items: list, background: BackgroundTasks) -> None:
    # делаем sync внутри background, чтобы webhook отвечал быстро
    for item in items:
        msg = extract_inbound(item)
        if not msg or not msg["chatId"]:
            continue

        chat_id_hash = chat_hash(msg["chatId"])

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

        if settings.BOT_SEND_ENABLED:
            await wazzup.send_message(
                chat_id=msg["chatId"],
                channel_id=msg["channelId"],
                chat_type=msg["chatType"],
                text=bot_reply.text,
            )


# ✅ ДВА маршрута: /webhooks и /webhook/wazzup
@app.post("/webhooks")
@app.post("/webhook/wazzup")
async def wazzup_webhook(request: Request, background: BackgroundTasks):
    _check_token(request)

    payload = await request.json()
    items = payload if isinstance(payload, list) else [payload]

    background.add_task(_process_items, items, background)
    return JSONResponse({"ok": True, "queued": len(items)})
