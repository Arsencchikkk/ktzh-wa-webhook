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
    ensure_session, load_active_case, load_active_case_by_type, create_case, update_case_with_message,
    required_slots, build_question, build_combined_question, format_dispatch_text, format_user_ack, close_case,
    set_active_case,
)

from .routing import resolve_region, resolve_executor

app = FastAPI(title="KTZH Smart Bot (Wazzup webhook)")
wazzup: Optional[WazzupClient] = None


# ----------------------------
# utils
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
# "AI" intent hints (rule-based but works well)
# ----------------------------
def _has_any(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)

COMPLAINT_WORDS = ["жалоб", "плох", "опозд", "гряз", "холод", "не работает", "ужас", "хам", "нет бумаги", "нет воды"]
LOST_WORDS = ["забыл", "оставил", "потерял", "утерял", "сумк", "рюкзак", "кошелек", "паспорт", "телефон", "вещ"]
GRAT_WORDS = ["благодар", "спасибо", "рахмет"]
GREET_WORDS = ["здравствуйте", "привет", "салам", "сәлем", "добрый", "ассалам"]


def detect_multi_intent(text: str, nlu_intent: str) -> Dict[str, bool]:
    t = (text or "").strip()
    want_greet = _has_any(t, GREET_WORDS)
    want_complaint = _has_any(t, COMPLAINT_WORDS)
    want_lost = _has_any(t, LOST_WORDS)
    want_grat = _has_any(t, GRAT_WORDS)

    # если greeting + complaint в одном — это НЕ greeting, а complaint
    if want_greet and (want_complaint or want_lost):
        want_greet = False

    # если nlu уже сказал complaint — оставим
    if nlu_intent == "complaint":
        want_complaint = True
    if nlu_intent == "lost_and_found":
        want_lost = True
    if nlu_intent == "gratitude":
        want_grat = True
    if nlu_intent == "greeting":
        want_greet = True

    return {"greet": want_greet, "complaint": want_complaint, "lost": want_lost, "grat": want_grat}


async def _send_text(channel_id: str, chat_type: str, chat_id: str, text: str, crm_id: str) -> None:
    if not settings.BOT_SEND_ENABLED:
        return
    if not wazzup:
        return
    await wazzup.send_text(
        channel_id=channel_id,
        chat_type=chat_type,
        chat_id=chat_id,
        text=text,
        crm_message_id=crm_id,
    )


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

    try:
        res = await wazzup.send_text(
            channel_id=channel_id,
            chat_type=chat_type,
            chat_id=chat_id,
            text=body.text,
            crm_message_id=f"debug-{int(time.time())}",
        )
        ok = getattr(res, "ok", None)
        if ok is None:
            ok = True
        return {
            "ok": ok,
            "used": {"channelId": channel_id, "chatType": chat_type, "chatId": chat_id},
            "BOT_SEND_ENABLED": settings.BOT_SEND_ENABLED,
            "HAS_WAZZUP_API_KEY": bool(settings.WAZZUP_API_KEY),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": repr(e),
            "BOT_SEND_ENABLED": settings.BOT_SEND_ENABLED,
            "HAS_WAZZUP_API_KEY": bool(settings.WAZZUP_API_KEY),
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

        dt = _safe_parse_dt(msg.get("dateTime"))
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

        # ✅ save message
        try:
            await m.messages.update_one(
                {"messageId": message_id},
                {"$setOnInsert": doc_insert, "$set": {k: v for k, v in doc_set.items() if v is not None}},
                upsert=True,
            )
        except Exception as e:
            print("❌ Mongo write error:", repr(e))
            return

        # bot logic only for inbound
        if direction != "inbound":
            continue
        if not (channel_id and chat_id and chat_type):
            continue
        if not _is_allowed_chat(chat_id):
            continue

        text = (msg.get("text") or "").strip()

        # session (context)
        sess = await ensure_session(m, channel_id, chat_id, chat_type)

        # NLU
        nlu = run_nlu(text) if text else run_nlu("")
        hints = detect_multi_intent(text, getattr(nlu, "intent", "other"))

        # greeting only
        if hints["greet"] and not (hints["complaint"] or hints["lost"] or hints["grat"]):
            greet = "Здравствуйте! Чем могу помочь? Можете написать жалобу или сообщение о забытых вещах."
            await _send_text(channel_id, chat_type, chat_id, greet, f"bot-greet-{message_id}")
            continue

        # handle "нашлась"
        active_any = await load_active_case(m, sess)
        if getattr(nlu, "is_found_message", False) and active_any and active_any.get("caseType") == "lost_and_found":
            await close_case(m, active_any["caseId"], status="closed")
            await _send_text(channel_id, chat_type, chat_id, "Отлично, рад(а) что нашлось! Если ещё нужна помощь — напишите.", f"bot-found-{message_id}")
            continue

        # --------
        # Multi-intent: complaint + lost in one message
        # --------
        if hints["complaint"] and hints["lost"]:
            # создаём/используем два активных кейса
            complaint_case = await load_active_case_by_type(m, sess, "complaint")
            lost_case = await load_active_case_by_type(m, sess, "lost_and_found")

            contact_name = None
            contact = msg.get("contact") or {}
            if isinstance(contact, dict):
                contact_name = contact.get("name")

            if not complaint_case:
                complaint_case = await create_case(
                    m, channel_id, chat_id, chat_type, contact_name, "complaint",
                    language=getattr(nlu, "language", "ru"),
                    seed_extracted={},
                )
                await set_active_case(m, channel_id, chat_id, "complaint", complaint_case["caseId"], make_primary=True)

            if not lost_case:
                lost_case = await create_case(
                    m, channel_id, chat_id, chat_type, contact_name, "lost_and_found",
                    language=getattr(nlu, "language", "ru"),
                    seed_extracted={},
                )
                await set_active_case(m, channel_id, chat_id, "lost_and_found", lost_case["caseId"], make_primary=False)

            # обновим оба кейса этим же сообщением (чтобы не терять контекст)
            complaint_case = await update_case_with_message(m, complaint_case, {**doc_set, "messageId": message_id}, nlu, sess=sess)
            lost_case = await update_case_with_message(m, lost_case, {**doc_set, "messageId": message_id}, nlu, sess=sess)

            miss_c = required_slots(complaint_case)
            miss_l = required_slots(lost_case)

            if miss_c or miss_l:
                q, slot = build_combined_question({"complaint": miss_c, "lost_and_found": miss_l})
                await _send_text(channel_id, chat_type, chat_id, "Понял(а) — вижу и жалобу, и забытые вещи. Сейчас оформлю оба обращения.\n" + q, f"bot-multi-q-{message_id}")
                # pendingSlot сохраняем один (упрощение)
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"pendingSlot": slot, "pendingQuestion": q, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue

            # если оба готовы — подтверждаем и закрываем/отправляем
            await _send_text(channel_id, chat_type, chat_id, f"{format_user_ack(complaint_case)}\n{format_user_ack(lost_case)}", f"bot-multi-ack-{message_id}")
            await close_case(m, complaint_case["caseId"], status="sent")
            await close_case(m, lost_case["caseId"], status="sent")
            continue

        # --------
        # Single intent flow (complaint / lost / gratitude / info)
        # --------
        case_type = getattr(nlu, "intent", "other")

        # нормализация: если одно слово "благодарность" — это gratitude, но НЕ закрываем
        if hints["grat"]:
            case_type = "gratitude"
        elif hints["lost"]:
            case_type = "lost_and_found"
        elif hints["complaint"]:
            case_type = "complaint"
        elif case_type == "greeting":
            # greeting уже обработали выше
            case_type = "other"

        # если "other" — но есть активный кейс, считаем это продолжением активного кейса
        case = await load_active_case(m, sess)
        if case_type == "other" and case:
            case_type = case.get("caseType", "other")

        # если совсем непонятно и нет кейса
        if case_type == "other" and not case:
            await _send_text(
                channel_id, chat_type, chat_id,
                "Уточните, пожалуйста: вы хотите подать жалобу или сообщить о забытых вещах?",
                f"bot-clarify-{message_id}"
            )
            continue

        # create/reuse case by type
        if not case or case.get("caseType") != case_type:
            case = await load_active_case_by_type(m, sess, case_type)

        if not case:
            contact_name = None
            contact = msg.get("contact") or {}
            if isinstance(contact, dict):
                contact_name = contact.get("name")

            case = await create_case(
                m, channel_id, chat_id, chat_type, contact_name, case_type,
                language=getattr(nlu, "language", "ru"),
                seed_extracted={},
            )
            await set_active_case(m, channel_id, chat_id, case_type, case["caseId"], make_primary=True)

        # update case with message (uses pendingSlot context)
        case = await update_case_with_message(m, case, {**doc_set, "messageId": message_id}, nlu, sess=sess)

        # Gratitude: не закрываем, пока нет текста благодарности
        if case.get("caseType") == "gratitude":
            miss = required_slots(case)
            if miss:
                q, slot = build_question(case, miss[0])
                await _send_text(channel_id, chat_type, chat_id, q, f"bot-grat-q-{message_id}")
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"pendingSlot": slot, "pendingQuestion": q, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue
            await _send_text(channel_id, chat_type, chat_id, format_user_ack(case), f"bot-grat-ack-{message_id}")
            await close_case(m, case["caseId"], status="closed")
            continue

        # Missing slots -> ask next question
        missing = required_slots(case)
        if missing:
            q, slot = build_question(case, missing[0])
            await _send_text(channel_id, chat_type, chat_id, q, f"bot-q-{case['caseId']}-{message_id}")
            await m.sessions.update_one(
                {"channelId": channel_id, "chatId": chat_id},
                {"$set": {"pendingSlot": slot, "pendingQuestion": q, "updatedAt": _now_utc()}},
                upsert=True,
            )
            continue

        # Dispatch / finalize
        if case.get("caseType") == "lost_and_found":
            if settings.LOST_FOUND_TARGET:
                await _send_text(
                    settings.LOST_FOUND_TARGET.get("channelId", channel_id) or channel_id,
                    settings.LOST_FOUND_TARGET.get("chatType", "whatsgroup"),
                    settings.LOST_FOUND_TARGET.get("chatId", chat_id),
                    format_dispatch_text(case),
                    f"bot-lf-{case['caseId']}",
                )
            await _send_text(channel_id, chat_type, chat_id, format_user_ack(case), f"bot-lf-ack-{message_id}")
            await close_case(m, case["caseId"], status="sent")
            continue

        if case.get("caseType") == "complaint":
            ex = case.get("extracted", {}) or {}
            region = resolve_region(ex.get("train"), ex.get("routeFrom"), ex.get("routeTo"))
            executor = resolve_executor(region)

            if executor.target_chat_id and executor.target_chat_type:
                await _send_text(
                    channel_id,
                    executor.target_chat_type,
                    executor.target_chat_id,
                    format_dispatch_text(case) + f"\nРегион: {executor.region}",
                    f"bot-complaint-{case['caseId']}",
                )

            await _send_text(channel_id, chat_type, chat_id, format_user_ack(case), f"bot-complaint-ack-{message_id}")
            await close_case(m, case["caseId"], status="sent")
            continue

        # fallback
        await _send_text(channel_id, chat_type, chat_id, format_user_ack(case), f"bot-ack-{message_id}")
        await close_case(m, case["caseId"], status="sent")

    # statuses (optional)
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
