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
    ensure_session,
    load_active_case,
    load_active_case_by_type,
    create_case,
    update_case_with_message,
    required_slots,
    build_question,
    format_dispatch_text,
    format_user_ack,
    close_case,
    set_active_case,
    set_pending,
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


async def _send_parts(channel_id: str, chat_type: str, chat_id: str, parts: List[str], crm_id: str) -> None:
    msg = "\n\n".join([p.strip() for p in parts if p and p.strip()])
    if msg:
        await _send_text(channel_id, chat_type, chat_id, msg, crm_id)


# ----------------------------
# “умные” подсказки (mini AI)
# ----------------------------
def _has_any(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def _looks_like_short_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # короткие ответы типа: "4", "вагон 8", "т58", "не помню"
    if t.isdigit() and 1 <= int(t) <= 9999:
        return True
    if len(t) <= 18 and any(x in t for x in ["вагон", "т", "t", "место", "не помню", "не знаю"]):
        return True
    return False


COMPLAINT_WORDS = ["жалоб", "плох", "опозд", "гряз", "холод", "не работает", "ужас", "хам", "нет бумаги", "нет воды"]
LOST_WORDS = ["забыл", "оставил", "потерял", "утерял", "сумк", "рюкзак", "кошелек", "паспорт", "телефон", "вещ"]
GRAT_WORDS = ["благодар", "спасибо", "рахмет"]
GREET_WORDS = ["здравствуйте", "привет", "салам", "сәлем", "добрый", "ассалам"]


def detect_intent_hints(text: str, nlu_intent: str) -> Dict[str, bool]:
    t = (text or "").strip()
    greet = _has_any(t, GREET_WORDS)
    complaint = _has_any(t, COMPLAINT_WORDS)
    lost = _has_any(t, LOST_WORDS)
    grat = _has_any(t, GRAT_WORDS)

    # greeting+complaint/lost => это не greeting
    if greet and (complaint or lost):
        greet = False

    # NLU override
    if nlu_intent == "complaint":
        complaint = True
    if nlu_intent == "lost_and_found":
        lost = True
    if nlu_intent == "gratitude":
        grat = True
    if nlu_intent == "greeting":
        greet = True

    return {"greet": greet, "complaint": complaint, "lost": lost, "grat": grat}


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

    try:
        res = await wazzup.send_text(
            channel_id=channel_id,
            chat_type=chat_type,
            chat_id=chat_id,
            text=body.text,
            crm_message_id=f"debug-{int(time.time())}",
        )
        ok = getattr(res, "ok", True)
        return {
            "ok": ok,
            "response": getattr(res, "response", None),
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

        # save message (idempotent)
        try:
            await m.messages.update_one(
                {"messageId": message_id},
                {"$setOnInsert": doc_insert, "$set": {k: v for k, v in doc_set.items() if v is not None}},
                upsert=True,
            )
        except Exception as e:
            print("❌ Mongo write error:", repr(e))
            return

        # bot only inbound
        if direction != "inbound":
            continue
        if not (channel_id and chat_id and chat_type):
            continue
        if not _is_allowed_chat(chat_id):
            continue

        text = (msg.get("text") or "").strip()

        sess = await ensure_session(m, channel_id, chat_id, chat_type)
        pending_case_type = sess.get("pendingCaseType")
        pending_slot = sess.get("pendingSlot")

        nlu = run_nlu(text) if text else run_nlu("")
        hints = detect_intent_hints(text, getattr(nlu, "intent", "other"))

        # ----------------------------
        # 1) Если ждём ответ (pending) и это похоже на короткий ответ — продолжаем нужный кейс
        # ----------------------------
        if pending_case_type and _looks_like_short_answer(text):
            if pending_case_type == "shared":
                # обновим все активные collecting/open кейсы (обычно complaint + lost)
                active_cases: List[Dict[str, Any]] = []
                for ct in ("complaint", "lost_and_found", "gratitude", "info"):
                    c = await load_active_case_by_type(m, sess, ct)
                    if c:
                        active_cases.append(c)

                for c in active_cases:
                    await update_case_with_message(m, c, {**doc_set, "messageId": message_id}, nlu, sess=sess)

                # после апдейта — задаём следующий вопрос/или завершаем
                # выберем приоритет: сначала train, потом carNumber (shared), потом остальное
                refreshed = [await load_active_case_by_type(m, sess, ct) for ct in ("complaint", "lost_and_found")]
                refreshed = [c for c in refreshed if c]

                if not refreshed:
                    continue

                # если train missing хотя бы где-то — спросим shared train
                if any("train" in required_slots(c) for c in refreshed):
                    q, slot = build_question(refreshed[0], "train")
                    await _send_text(channel_id, chat_type, chat_id, q, f"bot-q-shared-train-{message_id}")
                    await set_pending(m, channel_id, chat_id, q, slot, "shared")
                    continue

                # если carNumber missing хотя бы где-то — спросим shared car
                if any("carNumber" in required_slots(c) for c in refreshed):
                    q, slot = build_question(refreshed[0], "carNumber")
                    await _send_text(channel_id, chat_type, chat_id, q, f"bot-q-shared-car-{message_id}")
                    await set_pending(m, channel_id, chat_id, q, slot, "shared")
                    continue

                # дальше — пусть упадёт в обычную логику ниже
                # (не continue)

            else:
                # конкретный кейс
                case = await load_active_case_by_type(m, sess, pending_case_type)
                if not case:
                    case = await load_active_case(m, sess)

                if case:
                    case = await update_case_with_message(
                        m, case, {**doc_set, "messageId": message_id}, nlu, sess=sess
                    )

                    missing = required_slots(case)
                    if missing:
                        q, slot = build_question(case, missing[0])
                        await _send_text(channel_id, chat_type, chat_id, q, f"bot-q-{case['caseId']}-{message_id}")
                        await set_pending(m, channel_id, chat_id, q, slot, case.get("caseType"))
                        continue

                    # если кейс готов — завершаем (dispatch/ack)
                    await _finalize_one_case(channel_id, chat_type, chat_id, message_id, m, case)
                    continue

        # ----------------------------
        # 2) Greeting only
        # ----------------------------
        if hints["greet"] and not (hints["complaint"] or hints["lost"] or hints["grat"]):
            await _send_text(
                channel_id,
                chat_type,
                chat_id,
                "Здравствуйте! Чем могу помочь? Вы можете написать жалобу или сообщение о забытых вещах.",
                f"bot-greet-{message_id}",
            )
            continue

        # ----------------------------
        # 3) Determine case types (multi-intent)
        # ----------------------------
        case_types: List[str] = []
        if hints["complaint"]:
            case_types.append("complaint")
        if hints["lost"]:
            case_types.append("lost_and_found")
        if hints["grat"]:
            case_types.append("gratitude")

        # если ничего не распознали — либо продолжаем активный кейс, либо уточняем
        if not case_types:
            active = await load_active_case(m, sess)
            if active:
                case_types = [active.get("caseType", "other")]
            else:
                await _send_text(
                    channel_id,
                    chat_type,
                    chat_id,
                    "Уточните, пожалуйста: вы хотите подать жалобу, поблагодарить или сообщить о забытых вещах?",
                    f"bot-clarify-{message_id}",
                )
                continue

        # ----------------------------
        # 4) Load/create cases by type
        # ----------------------------
        contact_name = None
        contact = msg.get("contact") or {}
        if isinstance(contact, dict):
            contact_name = contact.get("name")

        cases: Dict[str, Dict[str, Any]] = {}
        for ct in case_types:
            c = await load_active_case_by_type(m, sess, ct)
            if not c:
                c = await create_case(
                    m,
                    channel_id=channel_id,
                    chat_id=chat_id,
                    chat_type=chat_type,
                    contact_name=contact_name,
                    case_type=ct,
                    language=getattr(nlu, "language", "ru"),
                    seed_extracted={},
                )
                await set_active_case(m, channel_id, chat_id, ct, c["caseId"], make_primary=True)
            cases[ct] = c

        # ----------------------------
        # 5) Apply this message to ALL involved cases (so context is shared)
        # ----------------------------
        for ct, c in list(cases.items()):
            cases[ct] = await update_case_with_message(
                m, c, {**doc_set, "messageId": message_id}, nlu, sess=sess
            )

        # ----------------------------
        # 6) Decide next step: shared questions first (train/car) if needed
        # ----------------------------
        collecting_cases = [c for c in cases.values() if c.get("status") in ("collecting", "open")]

        # если есть и complaint и lost — спрашиваем train/car один раз (shared)
        if ("complaint" in cases) and ("lost_and_found" in cases):
            # shared train
            if any("train" in required_slots(c) for c in (cases["complaint"], cases["lost_and_found"])):
                q, slot = build_question(cases["complaint"], "train")
                await _send_text(channel_id, chat_type, chat_id, q, f"bot-q-shared-train-{message_id}")
                await set_pending(m, channel_id, chat_id, q, slot, "shared")
                continue

            # shared car
            if any("carNumber" in required_slots(c) for c in (cases["complaint"], cases["lost_and_found"])):
                q, slot = build_question(cases["complaint"], "carNumber")
                await _send_text(channel_id, chat_type, chat_id, q, f"bot-q-shared-car-{message_id}")
                await set_pending(m, channel_id, chat_id, q, slot, "shared")
                continue

        # иначе — спрашиваем по одному самому важному кейсу
        # приоритет: lost_and_found (больше слотов) -> complaint -> gratitude
        priority = ["lost_and_found", "complaint", "gratitude", "info"]
        asked = False
        for ct in priority:
            if ct not in cases:
                continue
            c = cases[ct]
            missing = required_slots(c)
            if missing:
                q, slot = build_question(c, missing[0])
                await _send_text(channel_id, chat_type, chat_id, q, f"bot-q-{c['caseId']}-{message_id}")
                await set_pending(m, channel_id, chat_id, q, slot, c.get("caseType"))
                asked = True
                break
        if asked:
            continue

        # ----------------------------
        # 7) Finalize ready cases
        # ----------------------------
        # если готово несколько — отправим один "человеческий" ответ пользователю
        user_parts: List[str] = []

        # complaint finalize
        if "complaint" in cases:
            c = cases["complaint"]
            if not required_slots(c):
                await _dispatch_complaint(m, c)
                user_parts.append(format_user_ack(c))
                await close_case(m, c["caseId"], status="sent")

        # lost finalize
        if "lost_and_found" in cases:
            c = cases["lost_and_found"]
            if not required_slots(c):
                await _dispatch_lost(m, c)
                user_parts.append(format_user_ack(c))
                await close_case(m, c["caseId"], status="sent")

        # gratitude finalize
        if "gratitude" in cases:
            c = cases["gratitude"]
            if not required_slots(c):
                user_parts.append(format_user_ack(c))
                await close_case(m, c["caseId"], status="closed")

        if user_parts:
            await _send_parts(channel_id, chat_type, chat_id, user_parts, f"bot-final-{message_id}")
            await set_pending(m, channel_id, chat_id, None, None, None)
            continue

    # statuses
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


# ----------------------------
# dispatch helpers
# ----------------------------
async def _dispatch_lost(m, case: Dict[str, Any]) -> None:
    # отправка в LOST_FOUND_TARGET (группа/номер)
    if settings.LOST_FOUND_TARGET:
        await _send_text(
            channel_id=case["channelId"],
            chat_type=settings.LOST_FOUND_TARGET.get("chatType", "whatsgroup"),
            chat_id=settings.LOST_FOUND_TARGET.get("chatId", case["chatId"]),
            text=format_dispatch_text(case),
            crm_id=f"bot-lf-{case['caseId']}",
        )


async def _dispatch_complaint(m, case: Dict[str, Any]) -> None:
    ex = case.get("extracted", {}) or {}
    region = resolve_region(ex.get("train"), ex.get("routeFrom"), ex.get("routeTo"))
    executor = resolve_executor(region)

    if executor.target_chat_id and executor.target_chat_type:
        await _send_text(
            channel_id=case["channelId"],
            chat_type=executor.target_chat_type,
            chat_id=executor.target_chat_id,
            text=format_dispatch_text(case) + f"\nРегион: {executor.region}",
            crm_id=f"bot-complaint-{case['caseId']}",
        )


async def _finalize_one_case(channel_id: str, chat_type: str, chat_id: str, message_id: str, m, case: Dict[str, Any]) -> None:
    ct = case.get("caseType")
    if ct == "lost_and_found":
        if not required_slots(case):
            await _dispatch_lost(m, case)
            await _send_text(channel_id, chat_type, chat_id, format_user_ack(case), f"bot-lf-ack-{message_id}")
            await close_case(m, case["caseId"], status="sent")
            await set_pending(m, channel_id, chat_id, None, None, None)
        return

    if ct == "complaint":
        if not required_slots(case):
            await _dispatch_complaint(m, case)
            await _send_text(channel_id, chat_type, chat_id, format_user_ack(case), f"bot-complaint-ack-{message_id}")
            await close_case(m, case["caseId"], status="sent")
            await set_pending(m, channel_id, chat_id, None, None, None)
        return

    if ct == "gratitude":
        if not required_slots(case):
            await _send_text(channel_id, chat_type, chat_id, format_user_ack(case), f"bot-grat-ack-{message_id}")
            await close_case(m, case["caseId"], status="closed")
            await set_pending(m, channel_id, chat_id, None, None, None)
        return
