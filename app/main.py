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
from .nlu import run_nlu
from .dialog import (
    ensure_session, load_active_case, create_case, update_case_with_message,
    required_slots, build_question, format_dispatch_text, format_user_ack, close_case
)
from .routing import resolve_region, resolve_executor

app = FastAPI(title="KTZH Smart Bot (Wazzup webhook)")
wazzup: Optional[WazzupClient] = None


def utcnow() -> datetime:
    # ✅ aware UTC
    return datetime.now(timezone.utc)


def _safe_parse_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        dt = v
    else:
        try:
            dt = parser.isoparse(str(v))
        except Exception:
            return None

    # ✅ normalize to aware UTC
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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
    if not chat_id:
        return False
    return chat_id in settings.ALLOWED_CHAT_IDS


# ----------------------------
# DEBUG endpoints
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

    res = await wazzup.send_text(
        channel_id=channel_id,
        chat_type=chat_type,
        chat_id=chat_id,
        text=body.text,
        crm_message_id=f"debug-{int(time.time())}"
    )

    return {
        "ok": res.ok,
        "response": res.response,
        "used": {"channelId": channel_id, "chatType": chat_type, "chatId": chat_id},
        "BOT_SEND_ENABLED": settings.BOT_SEND_ENABLED,
        "HAS_WAZZUP_API_KEY": bool(settings.WAZZUP_API_KEY),
        "ALLOWED_CHAT_IDS": settings.ALLOWED_CHAT_IDS,
    }


@app.get("/debug/mongo")
async def debug_mongo(request: Request):
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    m = db_module.mongo
    if not m:
        raise HTTPException(status_code=503, detail="Mongo not ready")

    ping = await m.db.command("ping")
    count = await m.messages.count_documents({})
    return {
        "ok": True,
        "ping": ping,
        "messages_count": count,
        "db": settings.DB_NAME,
        "col": settings.COL_MESSAGES,
        "BOT_SEND_ENABLED": settings.BOT_SEND_ENABLED,
        "HAS_WAZZUP_API_KEY": bool(settings.WAZZUP_API_KEY),
        "ALLOWED_CHAT_IDS": settings.ALLOWED_CHAT_IDS,
    }


# ----------------------------
# lifecycle
# ----------------------------
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


@app.get("/")
async def root():
    return {"ok": True}


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
    m = db_module.mongo
    if m is None:
        print("❌ mongo is None (db_module.mongo). Check MONGODB_URI and init_mongo()")
        return

    try:
        messages: List[Dict[str, Any]] = payload.get("messages") or []
        statuses: List[Dict[str, Any]] = payload.get("statuses") or []

        # 1) store messages + bot logic
        for msg in messages:
            message_id = msg.get("messageId")
            if not message_id:
                continue

            channel_id = msg.get("channelId")
            chat_id = msg.get("chatId")
            chat_type = msg.get("chatType")
            is_echo = bool(msg.get("isEcho", False))
            direction = "outbound" if is_echo else "inbound"

            dt = _safe_parse_dt(msg.get("dateTime"))

            doc_insert = {"messageId": message_id, "createdAt": dt or utcnow()}

            # ❗️ВАЖНО: НЕ кладём messageId в $set (иначе конфликт как у тебя раньше)
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
                "updatedAt": utcnow(),
            }

            # ✅ always save in Mongo
            await m.messages.update_one(
                {"messageId": message_id},
                {"$setOnInsert": doc_insert, "$set": {k: v for k, v in doc_set.items() if v is not None}},
                upsert=True
            )

            # ✅ bot reacts only to inbound
            if direction != "inbound":
                continue
            if not (channel_id and chat_id and chat_type):
                continue

            # ✅ test mode: only your number
            if not _is_allowed_chat(chat_id):
                continue

            text = (msg.get("text") or "").strip()
            nlu = run_nlu(text) if text else run_nlu("")

            sess = await ensure_session(m, channel_id, chat_id, chat_type)
            active_case = await load_active_case(m, sess)

            # attachment without text -> attach to active case
            if msg.get("contentUri") and active_case:
                await update_case_with_message(m, active_case, {"messageId": message_id, **doc_set}, nlu)
                continue

            # greeting
            if nlu.intent == "greeting" and not active_case:
                greet = "Здравствуйте! Чем могу помочь? Напишите: жалоба / благодарность / забытая вещь."
                if nlu.language == "kk":
                    greet = "Сәлеметсіз бе! Қалай көмектесе аламын? Жазыңыз: шағым / алғыс / ұмытылған зат."
                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=greet, crm_message_id=f"bot-greet-{message_id}"
                    )
                continue

            # intent detection
            case_type = nlu.intent
            if case_type == "other":
                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text="Уточните, пожалуйста: жалоба / благодарность / забытая вещь?",
                        crm_message_id=f"bot-clarify-{message_id}"
                    )
                continue

            # create/reuse case
            case = active_case
            if not case:
                contact_name = None
                contact = msg.get("contact") or {}
                if isinstance(contact, dict):
                    contact_name = contact.get("name")
                case = await create_case(
                    m,
                    channel_id=channel_id,
                    chat_id=chat_id,
                    chat_type=chat_type,
                    contact_name=contact_name,
                    case_type=case_type,
                    nlu=nlu,
                )

            # update case evidence
            case = await update_case_with_message(m, case, {"messageId": message_id, **doc_set}, nlu)

            # gratitude fast
            if case.get("caseType") == "gratitude":
                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=format_user_ack(case),
                        crm_message_id=f"bot-gratitude-{message_id}"
                    )
                await close_case(m, case["caseId"], status="closed")
                continue

            # ask missing slots
            missing = required_slots(case)
            if missing:
                question = build_question(case, missing)
                if question and wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=question,
                        crm_message_id=f"bot-q-{case['caseId']}-{message_id}"
                    )
                continue

            dispatch_text = format_dispatch_text(case)

            # lost_and_found
            if case.get("caseType") == "lost_and_found":
                if settings.LOST_FOUND_TARGET and wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id,
                        chat_type=settings.LOST_FOUND_TARGET.get("chatType", "whatsgroup"),
                        chat_id=settings.LOST_FOUND_TARGET.get("chatId", chat_id),
                        text=dispatch_text,
                        crm_message_id=f"bot-lf-{case['caseId']}"
                    )
                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=format_user_ack(case),
                        crm_message_id=f"bot-lf-ack-{message_id}"
                    )
                await close_case(m, case["caseId"], status="sent")
                continue

            # complaint
            if case.get("caseType") == "complaint":
                ex = case.get("extracted", {})
                region = resolve_region(ex.get("train"), ex.get("routeFrom"), ex.get("routeTo"))
                executor = resolve_executor(region)

                if executor.target_chat_id and executor.target_chat_type and wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id,
                        chat_type=executor.target_chat_type,
                        chat_id=executor.target_chat_id,
                        text=dispatch_text + f"\nРегион: {executor.region}",
                        crm_message_id=f"bot-complaint-{case['caseId']}"
                    )

                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=format_user_ack(case),
                        crm_message_id=f"bot-complaint-ack-{message_id}"
                    )

                await close_case(m, case["caseId"], status="sent")
                continue

        # 2) statuses
        for st in statuses:
            mid = st.get("messageId")
            if not mid:
                continue
            await m.messages.update_one(
                {"messageId": mid},
                {"$set": {"currentStatus": st.get("status"), "statusRaw": st, "updatedAt": utcnow()}},
                upsert=True
            )

    except Exception as e:
        print("❌ process_payload crashed:", repr(e))
        return
