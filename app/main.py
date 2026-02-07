from __future__ import annotations
from typing import Any, Dict, Optional
import hashlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from .settings import settings
from .db import MongoStore
from .dialog import DialogManager
from .wazzup_client import WazzupClient


app = FastAPI(title=settings.APP_NAME)
store = MongoStore()
dialog = DialogManager(store)
wazzup = WazzupClient()


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


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/webhook/wazzup")
async def wazzup_webhook(request: Request):
    if settings.WEBHOOK_TOKEN:
        token = request.query_params.get("token") or request.query_params.get("crmKey")
        if token != settings.WEBHOOK_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid webhook token")

    payload = await request.json()
    items = payload if isinstance(payload, list) else [payload]

    replies = []
    for item in items:
        msg = extract_inbound(item)
        if not msg or not msg["chatId"]:
            continue

        chat_id_hash = chat_hash(msg["chatId"])
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

        if settings.BOT_SEND_ENABLED:
            send_res = wazzup.send_message(
                chat_id=msg["chatId"],
                channel_id=msg["channelId"],
                chat_type=msg["chatType"],
                text=bot_reply.text,
            )
        else:
            send_res = {"ok": True, "skipped": True}

        replies.append({"reply": bot_reply.text, "send": send_res})

    return JSONResponse({"ok": True, "handled": len(replies), "results": replies})
