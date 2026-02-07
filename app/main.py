from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from dateutil import parser
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import settings
from . import db as db_module
from .db import close_mongo, init_mongo
from .nlu import run_nlu
from .routing import resolve_executor, resolve_region
from .wazzup_client import WazzupClient
from .dialog import (
    build_question,
    close_case,
    create_case,
    ensure_session,
    format_dispatch_text,
    format_user_ack,
    load_active_case,
    load_active_case_by_type,
    required_slots,
    set_active_case,
    set_pending,
    update_case_with_message,
)

app = FastAPI(title="KTZH Smart Bot (Wazzup webhook)")
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
    if not chat_id:
        return False
    return chat_id in settings.ALLOWED_CHAT_IDS


async def _send_text(channel_id: str, chat_type: str, chat_id: str, text: str, crm_id: str) -> None:
    if not settings.BOT_SEND_ENABLED:
        return
    if not wazzup:
        return
    if not settings.WAZZUP_API_KEY:
        return
    try:
        await wazzup.send_text(
            channel_id=channel_id,
            chat_type=chat_type,
            chat_id=chat_id,
            text=text,
            crm_message_id=crm_id,
        )
    except Exception as e:
        print("❌ send_text error:", repr(e))


# ----------------------------
# scoring / intent
# ----------------------------
GREET_WORDS = ["здравствуйте", "привет", "салам", "сәлем", "добрый", "ассалам"]
THANK_WORDS = ["спасибо", "рахмет", "благодар", "алғыс","молодец","чист","уютн","комфорт"]
LOST_WORDS = ["забыл", "оставил", "потерял", "утерял", "вещ", "потерял"]
COMPLAINT_WORDS = [
    "жалоб", "плох", "ужас", "хам", "опозд", "задерж", "гряз", "вон", "холод", "жарк",
    "не работает", "сломал", "нет бумаги", "нет воды", "санитар", "шум", "течет","отсуствует"
]
STAFF_WORDS = ["проводник", "кондуктор", "кассир", "стюард", "начальник поезда","сотрудник"]


def _has_any(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def _meaning_score(text: str) -> int:
    t = (text or "").strip().lower()
    if not t:
        return 0
    if t in ("ок", "okay", "понял", "понятно", "да", "нет", "угу", "ага"):
        return 0
    if t.isdigit():
        return 0
    score = 0
    if len(t) >= 10:
        score += 1
    if len(t) >= 25:
        score += 1
    if _has_any(t, LOST_WORDS):
        score += 2
    if _has_any(t, COMPLAINT_WORDS):
        score += 2
    if _has_any(t, THANK_WORDS):
        score += 1
    return score


def _intent_scores(text: str, nlu_intent: str) -> Dict[str, int]:
    t_raw = (text or "").strip()
    t = t_raw.lower()

    has_greet = _has_any(t, GREET_WORDS)
    has_grat = _has_any(t, THANK_WORDS) or t in ("благодарность", "алғыс", "рахмет", "спасибо")
    has_lost = _has_any(t, LOST_WORDS)
    has_complaint = _has_any(t, COMPLAINT_WORDS)

    greet = 2 if has_greet else 0
    grat = 2 if has_grat else 0
    lost = 3 if has_lost else 0
    complaint = 3 if has_complaint else 0

    # greeting + (complaint/lost/grat) => это не greeting
    if greet and (lost or complaint or grat):
        greet = 0

    # NLU boosts (но ниже будет hard-override)
    if nlu_intent == "complaint":
        complaint += 2
    if nlu_intent == "lost_and_found":
        lost += 2
    if nlu_intent == "gratitude":
        grat += 2
    if nlu_intent == "greeting":
        greet += 1

 
    if has_grat and not has_complaint and not has_lost:
        grat = max(grat, 6)
        complaint = 0
        lost = 0
        greet = 0

    # Если это чистое приветствие (смысла 0) — НЕ создаём кейс
    if has_greet and _meaning_score(t_raw) == 0 and not (has_grat or has_lost or has_complaint):
        greet = max(greet, 6)
        grat = 0
        lost = 0
        complaint = 0

    # усиление благодарности если упомянут сотрудник
    if grat and _has_any(t, STAFF_WORDS):
        grat += 1

    return {"greet": greet, "gratitude": grat, "lost_and_found": lost, "complaint": complaint}



def _choose_case_types(text: str, nlu_intent: str, active_types: Optional[Set[str]] = None) -> List[str]:
    scores = _intent_scores(text, nlu_intent)
    active_types = active_types or set()

    # greeting only
    if scores["greet"] >= 5 and scores["complaint"] == 0 and scores["lost_and_found"] == 0 and scores["gratitude"] == 0:
        return ["greeting"]

 
    if scores["gratitude"] >= 5 and scores["complaint"] < 3 and scores["lost_and_found"] < 3:
        return ["gratitude"]

    picks: List[str] = []
    if scores["complaint"] >= 3:
        picks.append("complaint")
    if scores["lost_and_found"] >= 3:
        picks.append("lost_and_found")

    # gratitude — только если нет complaint/lost
    if not picks and scores["gratitude"] >= 3:
        picks.append("gratitude")

    # если нет picks, но есть активный контекст — продолжаем его
    if not picks and active_types:
        if "complaint" in active_types:
            return ["complaint"]
        if "lost_and_found" in active_types:
            return ["lost_and_found"]
        if "gratitude" in active_types:
            return ["gratitude"]

    if not picks:
        if nlu_intent in ("complaint", "lost_and_found", "gratitude"):
            picks.append(nlu_intent)
        else:
            picks.append("other")

    return picks



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
            "ALLOWED_CHAT_IDS": list(settings.ALLOWED_CHAT_IDS or []),
        }
    except Exception as e:
        return {"ok": False, "error": repr(e)}


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
        "ALLOWED_CHAT_IDS": list(settings.ALLOWED_CHAT_IDS or []),
    }


# ----------------------------
# lifecycle
# ----------------------------
@app.on_event("startup")
async def startup():
    global wazzup
    await init_mongo()
    wazzup = WazzupClient(settings.WAZZUP_API_KEY)
    try:
        await wazzup.start()
    except Exception as e:
        print("❌ wazzup start error:", repr(e))


@app.on_event("shutdown")
async def shutdown():
    global wazzup
    if wazzup:
        try:
            await wazzup.close()
        except Exception:
            pass
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


async def _ensure_case(m, sess: Dict[str, Any], channel_id: str, chat_id: str, chat_type: str, case_type: str, msg: Dict[str, Any], nlu: Any) -> Dict[str, Any]:
    case = await load_active_case_by_type(m, sess, case_type)
    if case:
        return case

    seed: Dict[str, Any] = {}
    any_active = await load_active_case(m, sess)
    if any_active:
        ex = any_active.get("extracted") or {}
        if ex.get("train"):
            seed["train"] = ex.get("train")
        if ex.get("carNumber"):
            seed["carNumber"] = ex.get("carNumber")

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
        language=getattr(nlu, "language", "ru"),
        seed_extracted=seed,
    )
    await set_active_case(m, channel_id, chat_id, case_type, case["caseId"], make_primary=True)
    return case


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

        try:
            await m.messages.update_one(
                {"messageId": message_id},
                {"$setOnInsert": doc_insert, "$set": {k: v for k, v in doc_set.items() if v is not None}},
                upsert=True,
            )
        except Exception as e:
            print("❌ Mongo write error:", repr(e))
            return

        if direction != "inbound":
            continue
        if not (channel_id and chat_id and chat_type):
            continue
        if not _is_allowed_chat(chat_id):
            continue

        text = (msg.get("text") or "").strip()
        nlu = run_nlu(text) if text else run_nlu("")

        sess = await ensure_session(m, channel_id, chat_id, chat_type)

        active_types: Set[str] = set()
        ac = sess.get("activeCases") or {}
        if isinstance(ac, dict):
            for k, v in ac.items():
                if v:
                    active_types.add(k)

        picked = _choose_case_types(text, getattr(nlu, "intent", "other"), active_types)

        if picked == ["greeting"]:
            greet = "Здравствуйте! Чем могу помочь? Можете написать жалобу, благодарность или сообщение о забытых вещах."
            await _send_text(channel_id, chat_type, chat_id, greet, f"bot-greet-{message_id}")
            continue

        if picked == ["other"] and not active_types:
            await _send_text(
                channel_id, chat_type, chat_id,
                "Уточните, пожалуйста: вы хотите подать жалобу, выразить благодарность или сообщить о забытых вещах?",
                f"bot-clarify-{message_id}",
            )
            continue

        cases: List[Dict[str, Any]] = []
        for ct in picked:
            if ct in ("complaint", "lost_and_found", "gratitude"):
                case = await _ensure_case(m, sess, channel_id, chat_id, chat_type, ct, msg, nlu)
                cases.append(case)

        if not cases:
            c = await load_active_case(m, sess)
            if c:
                cases = [c]

        msg_doc = {**doc_set, "messageId": message_id}
        updated_cases: List[Dict[str, Any]] = []
        for c in cases:
            updated = await update_case_with_message(m, c, msg_doc, nlu, sess=sess)
            updated_cases.append(updated)

        # ---- Универсальный LOST intake (если кейс ТОЛЬКО lost_and_found и не хватает базовых полей)
        if len(updated_cases) == 1 and updated_cases[0].get("caseType") == "lost_and_found":
            miss = required_slots(updated_cases[0])
            if "train" in miss or "carNumber" in miss or "seat" in miss or "when" in miss:
                q = (
                    "Понял(а). Чтобы быстрее найти вещь, напишите одним сообщением:\n"
                    "• номер поезда (Т58)\n"
                    "• номер вагона\n"
                    "• место (если помните, иначе «не помню»)\n"
                    "• что за вещь и приметы (цвет/бренд/что внутри)\n"
                    "• когда примерно оставили\n\n"
                    "Пример: «Т58, вагон 7, место 12, черная сумка Adidas, вчера 19:30»."
                )
                await _send_text(channel_id, chat_type, chat_id, q, f"bot-lf-intake-{message_id}")
                await set_pending(
                    m,
                    channel_id=channel_id,
                    chat_id=chat_id,
                    question=q,
                    slot="lf_intake",
                    targets=["lost_and_found"],
                    pending_case_type="lost_and_found",
                )
                continue

        # ---- SHARED train/car для complaint+lost (вместе)
        need_shared_train = False
        need_shared_car = False
        targets: List[str] = []
        for c in updated_cases:
            if c.get("caseType") in ("complaint", "lost_and_found"):
                ex = c.get("extracted") or {}
                if not ex.get("train"):
                    need_shared_train = True
                if not ex.get("carNumber"):
                    need_shared_car = True
                if c.get("caseType") not in targets:
                    targets.append(c.get("caseType"))

        if targets and (need_shared_train or need_shared_car):
            q = (
                "Уточните, пожалуйста, номер поезда и номер вагона (можно одним сообщением: «Т58, вагон 7»).\n"
                "Если это про забытые вещи — можете сразу добавить место и описание вещи."
            )
            await _send_text(channel_id, chat_type, chat_id, q, f"bot-q-shared-{message_id}")
            await set_pending(
                m,
                channel_id=channel_id,
                chat_id=chat_id,
                question=q,
                slot="train_car",
                targets=targets,
                pending_case_type="shared",
            )
            continue

        # ---- по одному вопросу (чтобы не спамить)
        asked = False
        for c in updated_cases:
            missing = required_slots(c)
            if missing:
                q, slot = build_question(c, missing[0])
                await _send_text(channel_id, chat_type, chat_id, q, f"bot-q-{c['caseId']}-{message_id}")
                await set_pending(
                    m,
                    channel_id=channel_id,
                    chat_id=chat_id,
                    question=q,
                    slot=slot,
                    targets=[c.get("caseType")],
                    pending_case_type=c.get("caseType"),
                )
                asked = True
                break

        if asked:
            continue

        # ---- dispatch/finalize
        for c in updated_cases:
            ct = c.get("caseType")

            if ct == "complaint":
                ex = c.get("extracted", {}) or {}
                region = resolve_region(ex.get("train"), ex.get("routeFrom"), ex.get("routeTo"))
                executor = resolve_executor(region)

                if executor.target_chat_id and executor.target_chat_type:
                    await _send_text(
                        channel_id=channel_id,
                        chat_type=executor.target_chat_type,
                        chat_id=executor.target_chat_id,
                        text=format_dispatch_text(c) + f"\nРегион: {executor.region}",
                        crm_id=f"bot-complaint-{c['caseId']}",
                    )

                await _send_text(channel_id, chat_type, chat_id, format_user_ack(c), f"bot-complaint-ack-{message_id}")
                await close_case(m, c["caseId"], status="sent")

            elif ct == "lost_and_found":
                if settings.LOST_FOUND_TARGET:
                    await _send_text(
                        channel_id=channel_id,
                        chat_type=settings.LOST_FOUND_TARGET.get("chatType", "whatsgroup"),
                        chat_id=settings.LOST_FOUND_TARGET.get("chatId", chat_id),
                        text=format_dispatch_text(c),
                        crm_id=f"bot-lf-{c['caseId']}",
                    )

                await _send_text(channel_id, chat_type, chat_id, format_user_ack(c), f"bot-lf-ack-{message_id}")
                await close_case(m, c["caseId"], status="sent")

            elif ct == "gratitude":
                await _send_text(channel_id, chat_type, chat_id, format_user_ack(c), f"bot-grat-ack-{message_id}")
                await close_case(m, c["caseId"], status="closed")

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
