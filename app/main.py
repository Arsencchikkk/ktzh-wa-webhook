from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dateutil import parser
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import settings
from .db import init_mongo, close_mongo
from . import db as db_module
from .wazzup_client import WazzupClient
from .dialog import handle_inbound_message


app = FastAPI(title="KTZH Smart Bot (Wazzup webhook + LLM)")
wazzup: Optional[WazzupClient] = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_parse_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        dt = parser.isoparse(str(v))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _hash_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    if not settings.PHONE_HASH_SALT:
        return phone
    raw = (settings.PHONE_HASH_SALT + phone).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _auth_ok(request: Request) -> bool:
    if settings.WEBHOOK_TOKEN:
        return (request.query_params.get("token") or "").strip() == settings.WEBHOOK_TOKEN
    return True


def _is_allowed_chat(chat_id: Optional[str]) -> bool:
    if not settings.ALLOWED_CHAT_IDS:
        return True
    return bool(chat_id) and chat_id in settings.ALLOWED_CHAT_IDS


async def _send_text(channel_id: str, chat_type: str, chat_id: str, text: str, crm_id: str) -> Dict[str, Any]:
    if not settings.BOT_SEND_ENABLED:
        return {"ok": True, "dryRun": True}

    if not wazzup:
        return {"ok": False, "error": "wazzup client not ready"}

    res = await wazzup.send_text(
        channel_id=channel_id,
        chat_type=chat_type,
        chat_id=chat_id,
        text=text,
        crm_message_id=crm_id,
    )
    return {"ok": res.ok, "response": res.response}


# ----------------------------
# debug
# ----------------------------
class DebugSend(BaseModel):
    text: str = "ping from bot"
    chat_id: Optional[str] = None
    chat_type: Optional[str] = None
    channel_id: Optional[str] = None


@app.post("/debug/send")
async def debug_send(request: Request, body: DebugSend):
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not wazzup:
        raise HTTPException(status_code=503, detail="Wazzup client not ready")

    channel_id = body.channel_id or settings.TEST_CHANNEL_ID
    chat_id = body.chat_id or settings.TEST_CHAT_ID
    chat_type = body.chat_type or settings.TEST_CHAT_TYPE

    if not (channel_id and chat_id and chat_type):
        raise HTTPException(status_code=400, detail="Need channel_id/chat_id/chat_type (or set TEST_* envs)")

    res = await _send_text(channel_id, chat_type, chat_id, body.text, f"debug-{int(time.time())}")
    return res


@app.get("/")
async def root():
    return {"ok": True}


@app.on_event("startup")
async def startup():
    global wazzup
    await init_mongo()
    wazzup = WazzupClient(settings.WAZZUP_API_KEY)
    await wazzup.start()


@app.on_event("shutdown")
async def shutdown():
    global wazzup
    if wazzup:
        await wazzup.close()
    await close_mongo()


# ----------------------------
# webhook
# ----------------------------
@app.post("/webhooks")
async def webhooks(request: Request, background: BackgroundTasks):
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": True}, status_code=200)

    if not isinstance(payload, dict):
        return JSONResponse({"ok": True}, status_code=200)

    if payload.get("test") is True:
        return JSONResponse({"ok": True}, status_code=200)

    background.add_task(process_payload, payload)
    return JSONResponse({"ok": True}, status_code=200)


async def process_payload(payload: Dict[str, Any]):
    """
    Wazzup webhooks: POST JSON, может приходить messages+statuses вместе,
    Authorization header может отсутствовать, важен ответ 200 OK. citeturn6view0
    """
    m = db_module.mongo
    if not m:
        print("❌ mongo is None")
        return

    messages: List[Dict[str, Any]] = payload.get("messages") or []
    statuses: List[Dict[str, Any]] = payload.get("statuses") or []

    for msg in messages:
        message_id = msg.get("messageId")
        if not message_id:
            continue

        channel_id = msg.get("channelId")
        chat_id = msg.get("chatId")
        chat_type = msg.get("chatType")

        is_echo = bool(msg.get("isEcho", False))
        direction = "outbound" if is_echo else "inbound"
        dt = _safe_parse_dt(msg.get("dateTime")) or _now_utc()

        doc_insert = {"messageId": message_id, "createdAt": dt}
        doc_set = {
            "channelId": channel_id,
            "chatId": chat_id,
            "chatIdHash": _hash_phone(chat_id),
            "chatType": chat_type,
            "dateTime": dt,
            "type": msg.get("type"),
            "direction": direction,
            "isEcho": is_echo,
            "text": msg.get("text"),
            "contentUri": msg.get("contentUri"),
            "contact": msg.get("contact"),
            "authorName": msg.get("authorName"),
            "authorId": msg.get("authorId"),
            "currentStatus": msg.get("status"),
            "raw": msg,
            "updatedAt": _now_utc(),
        }

        # save message idempotently
        res = await m.messages.update_one(
            {"messageId": message_id},
            {"$setOnInsert": doc_insert, "$set": {k: v for k, v in doc_set.items() if v is not None}},
            upsert=True,
        )

        # анти-дубль: если уже было — не отвечаем повторно
        if direction == "inbound" and not getattr(res, "upserted_id", None):
            continue

        if direction != "inbound":
            continue
        if not (channel_id and chat_id and chat_type):
            continue
        if not _is_allowed_chat(chat_id):
            continue

        contact_name = None
        if isinstance(msg.get("contact"), dict):
            contact_name = msg["contact"].get("name")

        await handle_inbound_message(
            m=m,
            send_text=_send_text,
            channel_id=channel_id,
            chat_id=chat_id,
            chat_type=chat_type,
            contact_name=contact_name,
            message_id=message_id,
            text=(msg.get("text") or ""),
            msg_type=(msg.get("type") or "text"),
            content_uri=msg.get("contentUri"),
            message_dt=dt,
        )

    for st in statuses:
        mid = st.get("messageId")
        if not mid:
            continue
        await m.messages.update_one(
            {"messageId": mid},
            {"$set": {"currentStatus": st.get("status"), "statusRaw": st, "updatedAt": _now_utc()}},
            upsert=True,
        )
