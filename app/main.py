from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import settings
from .db import init_mongo, close_mongo
from . import db as db_module  # IMPORTANT: use db_module.mongo (single global)
from .wazzup_client import WazzupClient
from .nlu import run_nlu
from .dialog import (
    ensure_session,
    load_active_case,
    create_case,
    update_case_with_message,
    required_slots,
    build_question,
    format_dispatch_text,
    format_user_ack,
    close_case,
)
from .routing import resolve_region, resolve_executor

app = FastAPI(title="KTZH Smart Bot (Wazzup webhook)")

wazzup: Optional[WazzupClient] = None


# ----------------------------
# time helpers (ALWAYS tz-aware)
# ----------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_parse_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        # ensure tz-aware
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        dt = parser.isoparse(str(v))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ----------------------------
# text heuristics (context understanding)
# ----------------------------
_GREET_ONLY = {
    "здравствуйте",
    "привет",
    "салам",
    "сәлем",
    "сәлеметсіз бе",
    "добрый день",
    "добрый вечер",
    "доброе утро",
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _is_greeting_only(text: str) -> bool:
    t = _norm(text)
    return t in _GREET_ONLY


def _contains_greeting(text: str) -> bool:
    t = _norm(text)
    return any(g in t for g in ["здравствуйте", "привет", "салам", "сәлем", "добрый"])


def _is_gratitude_only(text: str) -> bool:
    t = _norm(text)
    return t in {"благодарность", "алғыс", "рахмет", "спасибо"}


def _has_case_type_words(text: str) -> bool:
    t = _norm(text)
    return any(w in t for w in ["жалоб", "шағым", "благодар", "алғыс", "спасибо", "рахмет", "забыл", "ұмыт", "потер", "жоғалт", "вещ", "телефон", "паспорт"])


def _extract_train(text: str) -> Optional[str]:
    # T58 / Т58 / T-58
    t = (text or "").strip()
    m = re.search(r"\b([TТ])\s*[-]?\s*(\d{1,4}[A-Za-zА-Яа-яЁё]?)\b", t)
    if not m:
        return None
    return (m.group(1) + m.group(2)).upper().replace(" ", "").replace("-", "")


def _extract_wagon(text: str) -> Optional[int]:
    t = _norm(text)
    m = re.search(r"\bвагон\s*(\d{1,2})\b", t)
    if m:
        return int(m.group(1))
    # if only digits like "4"
    if t.isdigit():
        return int(t)
    return None


def _infer_intent(text: str, nlu_intent: str) -> str:
    """
    Heuristic override when NLU is weak:
    - greeting-only => greeting
    - greeting + complaint keywords => complaint
    - gratitude keywords => gratitude (but not auto-close)
    - lost keywords => lost_and_found
    """
    t = _norm(text)

    if _is_greeting_only(t):
        return "greeting"

    # explicit complaint signals
    if any(k in t for k in ["жалоб", "шағым", "ужас", "плохо", "холодно", "гряз", "санитар", "нет бумаги", "нет воды", "хам", "проблем", "не работает"]):
        return "complaint"

    # explicit lost&found signals
    if any(k in t for k in ["забыл", "потерял", "оставил", "ұмытып", "жоғалт", "lost", "forgot"]):
        return "lost_and_found"

    # gratitude signals
    if any(k in t for k in ["спасибо", "благодар", "рахмет", "алғыс"]):
        return "gratitude"

    return nlu_intent or "other"


# ----------------------------
# security & utils
# ----------------------------
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
    # if list empty -> allow all
    if not settings.ALLOWED_CHAT_IDS:
        return True
    if not chat_id:
        return False
    return chat_id in settings.ALLOWED_CHAT_IDS


async def _send_text(channel_id: str, chat_type: str, chat_id: str, text: str, crm_message_id: str) -> None:
    if not settings.BOT_SEND_ENABLED:
        return
    if not wazzup:
        return
    await wazzup.send_text(
        channel_id=channel_id,
        chat_type=chat_type,
        chat_id=chat_id,
        text=text,
        crm_message_id=crm_message_id,
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

    channel_id = body.channel_id or settings.TEST_CHANNEL_ID
    chat_id = body.chat_id or settings.TEST_CHAT_ID
    chat_type = body.chat_type or settings.TEST_CHAT_TYPE

    if not (channel_id and chat_id and chat_type):
        raise HTTPException(status_code=400, detail="Need channel_id/chat_id/chat_type (or set TEST_* envs)")

    if not settings.WAZZUP_API_KEY:
        return {
            "ok": False,
            "response": {"error": "WAZZUP_API_KEY is empty"},
            "used": {"channelId": channel_id, "chatType": chat_type, "chatId": chat_id},
            "BOT_SEND_ENABLED": settings.BOT_SEND_ENABLED,
            "HAS_WAZZUP_API_KEY": False,
        }

    if not wazzup:
        raise HTTPException(status_code=503, detail="Wazzup client not ready")

    res = await wazzup.send_text(
        channel_id=channel_id,
        chat_type=chat_type,
        chat_id=chat_id,
        text=body.text,
        crm_message_id=f"debug-{int(time.time())}",
    )

    # try to normalize response
    ok = getattr(res, "ok", None)
    response = getattr(res, "response", None)
    return {
        "ok": bool(ok) if ok is not None else True,
        "response": response,
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
    if settings.WAZZUP_API_KEY:
        wazzup = WazzupClient(settings.WAZZUP_API_KEY)
        await wazzup.start()
    else:
        wazzup = None


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
        print("❌ mongo is None. Check MONGODB_URI and init_mongo()")
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

            # store always
            try:
                await m.messages.update_one(
                    {"messageId": message_id},
                    {"$setOnInsert": doc_insert, "$set": {k: v for k, v in doc_set.items() if v is not None}},
                    upsert=True,
                )
            except Exception as e:
                print("❌ Mongo write error:", repr(e))
                continue

            # only handle inbound human messages
            if direction != "inbound":
                continue
            if not (channel_id and chat_id and chat_type):
                continue
            if not _is_allowed_chat(chat_id):
                continue

            text = (msg.get("text") or "").strip()

            # session + active case (context memory)
            sess = await ensure_session(m, channel_id, chat_id, chat_type)
            active_case = await load_active_case(m, sess)

            # attachments: add to active case and do not spam
            if msg.get("contentUri") and active_case:
                nlu = run_nlu("")  # no text
                await update_case_with_message(m, active_case, doc_set, nlu)
                continue

            # greeting-only (NO CASE)
            if _is_greeting_only(text) and not active_case:
                await _send_text(
                    channel_id, chat_type, chat_id,
                    "Здравствуйте! Чем могу помочь? Напишите: жалоба / благодарность / забытая вещь.",
                    f"bot-hello-{message_id}"
                )
                continue

            # NLU + heuristic override
            nlu = run_nlu(text) if text else run_nlu("")
            nlu_intent = getattr(nlu, "intent", "other") or "other"
            intent = _infer_intent(text, nlu_intent)

            # if no active case and user wrote train/wagon without type -> draftSlots
            train_hint = _extract_train(text)
            wagon_hint = _extract_wagon(text)

            if not active_case and (train_hint or wagon_hint is not None) and not _has_case_type_words(text):
                draft = (sess or {}).get("draftSlots") or {}
                if train_hint:
                    draft["train"] = train_hint
                if wagon_hint is not None:
                    draft["wagon"] = wagon_hint

                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"draftSlots": draft, "updatedAt": _now_utc()}},
                    upsert=True,
                )

                await _send_text(
                    channel_id, chat_type, chat_id,
                    "Понял(а). Это жалоба, благодарность или забытая вещь? Напишите одним словом и кратко суть.",
                    f"bot-clarify-type-{message_id}"
                )
                continue

            # If active case exists, DO NOT switch intent to "other" and DO NOT ask type again.
            case = active_case
            case_type = case.get("caseType") if case else intent

            # If no active case and intent still unknown -> ask type
            if not case and case_type in (None, "", "other", "greeting"):
                await _send_text(
                    channel_id, chat_type, chat_id,
                    "Уточните, пожалуйста: жалоба / благодарность / забытая вещь?",
                    f"bot-clarify-{message_id}"
                )
                continue

            # Merge draft slots into nlu.slots when starting case
            if not case:
                draft = (sess or {}).get("draftSlots") or {}
                try:
                    slots = getattr(nlu, "slots", None)
                    if slots is None:
                        setattr(nlu, "slots", {})
                        slots = nlu.slots
                    if isinstance(slots, dict) and draft:
                        for k, v in draft.items():
                            slots.setdefault(k, v)
                except Exception:
                    pass

                # clear draft
                if (sess or {}).get("draftSlots"):
                    await m.sessions.update_one(
                        {"channelId": channel_id, "chatId": chat_id},
                        {"$set": {"draftSlots": {}, "updatedAt": _now_utc()}},
                        upsert=True,
                    )

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

            # update case with this message and apply pendingSlot answers
            case = await update_case_with_message(m, case, doc_set, nlu)

            # Determine missing slots & ask next question
            missing = required_slots(case)

            if missing:
                slot_key = missing[0]
                question_text = build_question(case, missing)

                # avoid repeating same question
                sess2 = await m.sessions.find_one({"channelId": channel_id, "chatId": chat_id})
                prev_slot = (sess2 or {}).get("pendingSlot")
                prev_text = (sess2 or {}).get("pendingQuestion")

                if slot_key != prev_slot or question_text != prev_text:
                    await _send_text(
                        channel_id, chat_type, chat_id,
                        question_text,
                        f"bot-q-{case['caseId']}-{message_id}"
                    )

                    await m.sessions.update_one(
                        {"channelId": channel_id, "chatId": chat_id},
                        {"$set": {"pendingSlot": slot_key, "pendingQuestion": question_text, "updatedAt": _now_utc()}},
                        upsert=True,
                    )
                continue

            # Ready to dispatch / finalize
            dispatch_text = format_dispatch_text(case)

            # gratitude: close only after details exist (required_slots already ensured)
            if case.get("caseType") == "gratitude":
                await _send_text(
                    channel_id, chat_type, chat_id,
                    "Спасибо за подробности! Передадим вашу благодарность команде 🙏",
                    f"bot-gratitude-done-{message_id}",
                )
                await close_case(m, case["caseId"], status="closed")
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"activeCaseId": None, "pendingSlot": None, "pendingQuestion": None, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue

            # info -> support target (optional)
            if case.get("caseType") == "info":
                if settings.SUPPORT_TARGET:
                    await _send_text(
                        channel_id=channel_id,
                        chat_type=settings.SUPPORT_TARGET.get("chatType", "whatsapp"),
                        chat_id=settings.SUPPORT_TARGET.get("chatId", chat_id),
                        text=dispatch_text,
                        crm_message_id=f"bot-support-{case['caseId']}",
                    )

                await _send_text(
                    channel_id, chat_type, chat_id,
                    format_user_ack(case),
                    f"bot-info-ack-{message_id}"
                )
                await close_case(m, case["caseId"], status="sent")
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"activeCaseId": None, "pendingSlot": None, "pendingQuestion": None, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue

            # lost_and_found -> lost&found target (group or phone)
            if case.get("caseType") == "lost_and_found":
                if settings.LOST_FOUND_TARGET:
                    await _send_text(
                        channel_id=channel_id,
                        chat_type=settings.LOST_FOUND_TARGET.get("chatType", "whatsgroup"),
                        chat_id=settings.LOST_FOUND_TARGET.get("chatId", chat_id),
                        text=dispatch_text,
                        crm_message_id=f"bot-lf-{case['caseId']}",
                    )

                await _send_text(
                    channel_id, chat_type, chat_id,
                    format_user_ack(case),
                    f"bot-lf-ack-{message_id}"
                )

                await close_case(m, case["caseId"], status="sent")
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"activeCaseId": None, "pendingSlot": None, "pendingQuestion": None, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue

            # complaint -> executor by region
            if case.get("caseType") == "complaint":
                ex = case.get("extracted", {}) or {}
                region = resolve_region(ex.get("train"), ex.get("routeFrom"), ex.get("routeTo"))
                executor = resolve_executor(region)

                if executor.target_chat_id and executor.target_chat_type:
                    await _send_text(
                        channel_id=channel_id,
                        chat_type=executor.target_chat_type,
                        chat_id=executor.target_chat_id,
                        text=dispatch_text + f"\nРегион: {executor.region}",
                        crm_message_id=f"bot-complaint-{case['caseId']}",
                    )

                await _send_text(
                    channel_id, chat_type, chat_id,
                    format_user_ack(case),
                    f"bot-complaint-ack-{message_id}"
                )

                await close_case(m, case["caseId"], status="sent")
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"activeCaseId": None, "pendingSlot": None, "pendingQuestion": None, "updatedAt": _now_utc()}},
                    upsert=True,
                )
                continue

        # 2) store statuses for analytics
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

    except Exception as e:
        print("❌ process_payload crashed:", repr(e))
        return
