from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dateutil import parser
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import settings
from .db import init_mongo, close_mongo
from . import db as db_module  # ✅ ВАЖНО: берем mongo через модуль, не через mongo_global
from .wazzup_client import WazzupClient
from .nlu import run_nlu
from .dialog import (
    ensure_session, load_active_case, create_case, update_case_with_message,
    required_slots, build_question, format_dispatch_text, format_user_ack, close_case
)
from .routing import resolve_region, resolve_executor

app = FastAPI(title="KTZH Smart Bot (Wazzup webhook)")
wazzup: Optional[WazzupClient] = None


# ----------------------------
# utils
# ----------------------------
def _safe_parse_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return parser.isoparse(str(v))
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
    # ✅ если список пуст — разрешаем всем
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

    await wazzup.send_text(
        channel_id=channel_id,
        chat_type=chat_type,
        chat_id=chat_id,
        text=body.text,
        crm_message_id=f"debug-{int(time.time())}"
    )
    return {"ok": True}

@app.get("/debug/mongo")
async def debug_mongo(request: Request):
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    m = db_module.mongo  # ✅ ВАЖНО
    if not m:
        raise HTTPException(status_code=503, detail="Mongo not ready")

    ping = await m.db.command("ping")
    count = await m.messages.count_documents({})
    return {"ok": True, "ping": ping, "messages_count": count, "db": settings.DB_NAME, "col": settings.COL_MESSAGES}


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
    m = db_module.mongo  # ✅ ВАЖНО
    if m is None:
        print("❌ mongo is None (db_module.mongo). Check MONGODB_URI and init_mongo()")
        return

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

        doc_insert = {"messageId": message_id, "createdAt": dt or datetime.utcnow()}

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
            "updatedAt": datetime.utcnow(),
        }

        # ✅ всегда сохраняем в Mongo
        try:
            await m.messages.update_one(
                {"messageId": message_id},
                {"$setOnInsert": doc_insert, "$set": {k: v for k, v in doc_set.items() if v is not None}},
                upsert=True
            )
        except Exception as e:
            print("❌ Mongo write error:", repr(e))
            return

        # ✅ бот отвечает только на inbound
        if direction != "inbound":
            continue
        if not (channel_id and chat_id and chat_type):
            continue

        # ✅ тестовый режим: отвечает только разрешенным chatId (твой номер)
        if not _is_allowed_chat(chat_id):
            continue

        text = (msg.get("text") or "").strip()
        nlu = run_nlu(text) if text else run_nlu("")

        sess = await ensure_session(m, channel_id, chat_id, chat_type)
        active_case = await load_active_case(m, sess)

        # ✅ если пришло вложение без текста — прикрепляем к активному кейсу и НЕ спамим вопросами
        if msg.get("contentUri") and active_case:
            await update_case_with_message(m, active_case, doc_set, nlu)
            continue

        # greeting (без активного кейса)
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

        # "нашлась" закрывает lost&found
        if nlu.is_found_message and active_case and active_case.get("caseType") == "lost_and_found":
            await close_case(m, active_case["caseId"], status="closed")
            if wazzup:
                await wazzup.send_text(
                    channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                    text="Отлично, рад(а) что нашлось! Если ещё нужна помощь — напишите.",
                    crm_message_id=f"bot-found-{message_id}"
                )
            continue

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

        # update case with this message
        case = await update_case_with_message(m, case, doc_set, nlu)

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

        # ask missing slots (one question)
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

        # dispatch
        dispatch_text = format_dispatch_text(case)

        # lost_and_found -> group/target
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

        # complaint -> executor
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

    # 2) store statuses
    for st in statuses:
        mid = st.get("messageId")
        if not mid:
            continue
        try:
            await m.messages.update_one(
                {"messageId": mid},
                {"$set": {"currentStatus": st.get("status"), "statusRaw": st, "updatedAt": datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            print("❌ Mongo status write error:", repr(e))
            return
