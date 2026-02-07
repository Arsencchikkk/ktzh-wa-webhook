from __future__ import annotations

import hashlib
import re
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


# ----------------------------
# time utils (ALWAYS tz-aware)
# ----------------------------
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


# ----------------------------
# auth / allowlist
# ----------------------------
def _auth_ok(request: Request) -> bool:
    if settings.WEBHOOK_TOKEN:
        return (request.query_params.get("token") or "").strip() == settings.WEBHOOK_TOKEN
    return True

def _is_allowed_chat(chat_id: Optional[str]) -> bool:
    # если список пуст — разрешаем всем
    if not settings.ALLOWED_CHAT_IDS:
        return True
    if not chat_id:
        return False
    return chat_id in settings.ALLOWED_CHAT_IDS


# ----------------------------
# hashing
# ----------------------------
def _hash_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    if not settings.PHONE_HASH_SALT:
        return phone
    raw = (settings.PHONE_HASH_SALT + phone).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ----------------------------
# greeting helpers (human-like)
# ----------------------------
_GREET_PREFIX_RX = re.compile(r"^\s*(здравствуй(те)?|привет|добрый\s+день|добрый\s+вечер|сәлем(етсіз\s*бе)?|сәлем)\b", re.IGNORECASE)

def _has_greeting_prefix(text: str) -> bool:
    return bool(_GREET_PREFIX_RX.search(text or ""))

def _hello_text(lang: str = "ru") -> str:
    if lang == "kk":
        return "Сәлеметсіз бе! Қалай көмектесе аламын? Жазыңыз: шағым / алғыс / ұмытылған зат."
    return "Здравствуйте! Чем могу помочь? Напишите: жалоба / благодарность / забытая вещь."


# ----------------------------
# CONTEXT: interpret short replies by pendingQuestion
# ----------------------------
def _apply_context_slots(text: str, sess: Dict[str, Any], nlu) -> None:
    """
    Если пользователь отвечает коротко (например: "4"),
    подставляем значения в слоты по last asked question (pendingQuestion).
    """
    t = (text or "").strip()
    if not t:
        return

    pq = (sess or {}).get("pendingQuestion") or ""
    pql = pq.lower()

    # 1) digit-only reply
    if t.isdigit():
        if "вагон" in pql:
            try:
                nlu.slots["wagon"] = int(t)
            except Exception:
                nlu.slots["wagon"] = t
        elif "мест" in pql:  # если потом добавишь seat
            nlu.slots["seat"] = t

    # 2) train reply when bot asked train/route
    if ("поезд" in pql or "маршрут" in pql) and not nlu.slots.get("train"):
        # accept: "т58", "T58", "58A"
        m = re.match(r"^\s*([TТ]?\s*\d{1,4}[A-Za-zА-Яа-яЁё]?)\s*$", t)
        if m:
            val = m.group(1).upper().replace(" ", "")
            nlu.slots["train"] = val

    # 3) date/time reply when bot asked date/time
    if ("дат" in pql or "врем" in pql) and (not nlu.slots.get("date") and not nlu.slots.get("time")):
        n2 = run_nlu(t)
        if n2.slots.get("date"):
            nlu.slots["date"] = n2.slots["date"]
        if n2.slots.get("time"):
            nlu.slots["time"] = n2.slots["time"]

    # 4) item reply when bot asked what item
    if ("что" in pql or "предмет" in pql or "забы" in pql or "потер" in pql) and not nlu.slots.get("item"):
        nlu.slots["item"] = t


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
        crm_message_id=f"debug-{int(time.time())}",
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

            # IMPORTANT: messageId is ONLY in filter and $setOnInsert (not in $set) to avoid conflict
            doc_insert = {"messageId": message_id, "createdAt": dt or _now_utc()}

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

            # ✅ always write message to Mongo
            try:
                await m.messages.update_one(
                    {"messageId": message_id},
                    {"$setOnInsert": doc_insert, "$set": {k: v for k, v in doc_set.items() if v is not None}},
                    upsert=True,
                )
            except Exception as e:
                print("❌ Mongo write error:", repr(e))
                return

            # bot triggers only for inbound non-echo
            if direction != "inbound":
                continue
            if not (channel_id and chat_id and chat_type):
                continue
            if not _is_allowed_chat(chat_id):
                continue

            text = (msg.get("text") or "").strip()
            nlu = run_nlu(text) if text else run_nlu("")

            sess = await ensure_session(m, channel_id, chat_id, chat_type)
            active_case = await load_active_case(m, sess)

            # ✅ attachments without text -> attach to active case (no spam)
            if msg.get("contentUri") and active_case and not text:
                await update_case_with_message(m, active_case, {"messageId": message_id, **doc_set}, nlu)
                continue

            # ✅ greeting ONLY + no active case -> just greet like a real person
            if nlu.intent == "greeting" and not active_case:
                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=_hello_text(nlu.language),
                        crm_message_id=f"bot-hello-{message_id}",
                    )
                continue

            # ✅ if there IS active case -> ANY message is continuation (context!)
            if active_case:
                # apply context mapping by pendingQuestion (digit reply etc.)
                _apply_context_slots(text, sess, nlu)

                case = await update_case_with_message(m, active_case, {"messageId": message_id, **doc_set}, nlu)

                # if user says just "здравствуйте" during active case -> politely continue with next question
                if nlu.intent == "greeting":
                    missing = required_slots(case)
                    if missing:
                        q = build_question(case, missing)
                        if q and wazzup:
                            pref = "Здравствуйте! " if nlu.language != "kk" else "Сәлеметсіз бе! "
                            await wazzup.send_text(
                                channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                                text=pref + q,
                                crm_message_id=f"bot-continue-{case['caseId']}-{message_id}",
                            )
                            await m.sessions.update_one(
                                {"channelId": channel_id, "chatId": chat_id},
                                {"$set": {"pendingQuestion": q, "updatedAt": _now_utc()}},
                                upsert=True,
                            )
                    else:
                        if wazzup:
                            msg_txt = "Здравствуйте! Я вижу вашу заявку, сейчас передам ответственным."
                            if nlu.language == "kk":
                                msg_txt = "Сәлеметсіз бе! Өтінішіңіз қабылданды, жауапты бөлімге жіберемін."
                            await wazzup.send_text(
                                channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                                text=msg_txt,
                                crm_message_id=f"bot-continue2-{case['caseId']}-{message_id}",
                            )
                    continue

            else:
                # ✅ no active case: handle "other" nicely
                case_type = nlu.intent
                if case_type == "other":
                    if wazzup:
                        txt = "Понял. Это жалоба, благодарность или забытая вещь?"
                        if nlu.language == "kk":
                            txt = "Түсіндім. Бұл шағым ба, алғыс па, әлде ұмытылған зат па?"
                        await wazzup.send_text(
                            channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                            text=txt,
                            crm_message_id=f"bot-clarify-{message_id}",
                        )
                    continue

                # create case
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
                case = await update_case_with_message(m, case, {"messageId": message_id, **doc_set}, nlu)

            # ---- from here case ALWAYS exists ----

            # gratitude fast
            if case.get("caseType") == "gratitude":
                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=format_user_ack(case),
                        crm_message_id=f"bot-gratitude-{message_id}",
                    )
                await close_case(m, case["caseId"], status="closed")
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"activeCaseId": None, "pendingQuestion": None, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue

            # info -> (optional) forward + ack + close
            if case.get("caseType") == "info":
                if settings.SUPPORT_TARGET and wazzup:
                    txt = format_dispatch_text(case)
                    await wazzup.send_text(
                        channel_id=channel_id,
                        chat_type=settings.SUPPORT_TARGET.get("chatType", "whatsapp"),
                        chat_id=settings.SUPPORT_TARGET.get("chatId", chat_id),
                        text=txt,
                        crm_message_id=f"bot-support-{case['caseId']}",
                    )
                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=format_user_ack(case),
                        crm_message_id=f"bot-info-ack-{message_id}",
                    )
                await close_case(m, case["caseId"], status="sent")
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"activeCaseId": None, "pendingQuestion": None, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue

            # ask missing slots (ONE question)
            missing = required_slots(case)
            if missing:
                q = build_question(case, missing)

                # don't repeat same question
                prev_q = (sess or {}).get("pendingQuestion")
                if q and q != prev_q and wazzup:
                    # human-like: if message started with greeting, soften response
                    pref = "Понял(а). " if not _has_greeting_prefix(text) else "Здравствуйте! "
                    if nlu.language == "kk":
                        pref = "Түсіндім. " if not _has_greeting_prefix(text) else "Сәлеметсіз бе! "

                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=pref + q,
                        crm_message_id=f"bot-q-{case['caseId']}-{message_id}",
                    )
                    await m.sessions.update_one(
                        {"channelId": channel_id, "chatId": chat_id},
                        {"$set": {"pendingQuestion": q, "updatedAt": _now_utc()}},
                        upsert=True,
                    )
                continue

            # dispatch ready
            dispatch_text = format_dispatch_text(case)

            if case.get("caseType") == "lost_and_found":
                # send to target (or skip)
                if settings.LOST_FOUND_TARGET and wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id,
                        chat_type=settings.LOST_FOUND_TARGET.get("chatType", "whatsgroup"),
                        chat_id=settings.LOST_FOUND_TARGET.get("chatId", chat_id),
                        text=dispatch_text,
                        crm_message_id=f"bot-lf-{case['caseId']}",
                    )
                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=format_user_ack(case),
                        crm_message_id=f"bot-lf-ack-{message_id}",
                    )
                await close_case(m, case["caseId"], status="sent")
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"activeCaseId": None, "pendingQuestion": None, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue

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
                        crm_message_id=f"bot-complaint-{case['caseId']}",
                    )

                if wazzup:
                    await wazzup.send_text(
                        channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                        text=format_user_ack(case),
                        crm_message_id=f"bot-complaint-ack-{message_id}",
                    )

                await close_case(m, case["caseId"], status="sent")
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"activeCaseId": None, "pendingQuestion": None, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue

        # 2) statuses
        for st in statuses:
            mid = st.get("messageId")
            if not mid:
                continue
            try:
                await m.messages.update_one(
                    {"messageId": mid},
                    {"$set": {"currentStatus": st.get("status"), "statusRaw": st, "updatedAt": _now_utc()}},
                    upsert=True,
                )
            except Exception as e:
                print("❌ Mongo status write error:", repr(e))
                return

    except Exception as e:
        print("❌ process_payload crashed:", repr(e))
        return
