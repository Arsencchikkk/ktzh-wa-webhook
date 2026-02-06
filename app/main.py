from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import settings
from .db import init_mongo, close_mongo, mongo as mongo_global
from .wazzup_client import WazzupClient
from .nlu import run_nlu
from .dialog import (
    ensure_session, load_active_case, create_case, update_case_with_message,
    required_slots, build_question, format_dispatch_text, format_user_ack, close_case
)
from .routing import resolve_region, resolve_executor

app = FastAPI(title="KTZH Smart Bot (Wazzup webhook)")

wazzup: Optional[WazzupClient] = None


def _hash_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    if not settings.PHONE_HASH_SALT:
        return phone
    raw = (settings.PHONE_HASH_SALT + phone).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _auth_ok(request: Request) -> bool:
    # Самое надёжное: token в query string (Authorization может не быть в webhooks)
    if settings.WEBHOOK_TOKEN:
        return (request.query_params.get("token") or "").strip() == settings.WEBHOOK_TOKEN
    return True


def _is_allowed_chat(chat_id: Optional[str]) -> bool:
    # Если задан ALLOWED_CHAT_IDS — бот отвечает ТОЛЬКО им (твой режим теста)
    if not settings.ALLOWED_CHAT_IDS:
        return True
    if not chat_id:
        return False
    return chat_id in settings.ALLOWED_CHAT_IDS


# ----------------------------
# ✅ DEBUG endpoint: принудительно отправить сообщение на твой номер
# ----------------------------
class DebugSend(BaseModel):
    text: str = "ping from bot"


@app.post("/debug/send")
async def debug_send(request: Request, body: DebugSend):
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not wazzup:
        raise HTTPException(status_code=503, detail="Wazzup client not ready")

    # Отправляем ТОЛЬКО на тестовый чат (чтобы случайно не ушло куда-то)
    if not (settings.TEST_CHANNEL_ID and settings.TEST_CHAT_ID and settings.TEST_CHAT_TYPE):
        raise HTTPException(
            status_code=400,
            detail="Set TEST_CHANNEL_ID / TEST_CHAT_ID / TEST_CHAT_TYPE in env (or settings.py)"
        )

    await wazzup.send_text(
        channel_id=settings.TEST_CHANNEL_ID,
        chat_type=settings.TEST_CHAT_TYPE,
        chat_id=settings.TEST_CHAT_ID,
        text=body.text,
        crm_message_id=f"debug-{int(time.time())}"
    )
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


@app.get("/")
async def root():
    return {"ok": True}


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

    # test webhook from Wazzup
    if payload.get("test") is True:
        return JSONResponse({"ok": True}, status_code=200)

    background.add_task(process_payload, payload)
    return JSONResponse({"ok": True}, status_code=200)


async def process_payload(payload: Dict[str, Any]):
    m = mongo_global
    if m is None:
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

        doc_insert = {
            "messageId": message_id,
            "createdAt": msg.get("dateTime"),
        }

        doc_set = {
            "messageId": message_id,
            "channelId": channel_id,
            "chatId": chat_id,
            "chatIdHash": _hash_phone(chat_id),
            "chatType": chat_type,
            "dateTime": msg.get("dateTime"),
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
        }

        # Upsert message
        await m.messages.update_one(
            {"messageId": message_id},
            {"$setOnInsert": doc_insert, "$set": {k: v for k, v in doc_set.items() if v is not None}},
            upsert=True
        )

        # Бот отвечает только на inbound и только на разрешённые чаты (твой тестовый номер)
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
            if msg.get("contentUri") and active_case:
                await update_case_with_message(m, active_case, doc_set, nlu)
                continue

            if wazzup:
                await wazzup.send_text(
                    channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                    text="Уточните, пожалуйста, что случилось: жалоба / благодарность / забытая вещь?",
                    crm_message_id=f"bot-clarify-{message_id}"
                )
            continue

        # normalize
        if case_type == "complaint":
            case_type = "complaint"
        elif case_type == "gratitude":
            case_type = "gratitude"
        elif case_type == "lost_and_found":
            case_type = "lost_and_found"
        elif case_type == "info":
            case_type = "info"

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
        else:
            if case.get("caseType") in (None, "", "other"):
                await m.cases.update_one({"caseId": case["caseId"]}, {"$set": {"caseType": case_type}})

        case = await update_case_with_message(m, case, doc_set, nlu)

        if case.get("caseType") == "gratitude":
            if wazzup:
                await wazzup.send_text(
                    channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                    text="Спасибо за обратную связь! Передадим благодарность команде 🙏",
                    crm_message_id=f"bot-gratitude-{message_id}"
                )
            await close_case(m, case["caseId"], status="closed")
            continue

        if case.get("caseType") == "info":
            if settings.SUPPORT_TARGET and wazzup:
                txt = format_dispatch_text(case)
                await wazzup.send_text(
                    channel_id=channel_id,
                    chat_type=settings.SUPPORT_TARGET.get("chatType", "whatsapp"),
                    chat_id=settings.SUPPORT_TARGET.get("chatId", chat_id),
                    text=txt,
                    crm_message_id=f"bot-support-{case['caseId']}"
                )
            if wazzup:
                await wazzup.send_text(
                    channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                    text=format_user_ack(case),
                    crm_message_id=f"bot-info-ack-{message_id}"
                )
            await close_case(m, case["caseId"], status="sent")
            continue

        missing = required_slots(case)
        if missing:
            question = build_question(case, missing)
            sess = await m.sessions.find_one({"channelId": channel_id, "chatId": chat_id})
            prev_q = (sess or {}).get("pendingQuestion")

            if question and question != prev_q and wazzup:
                await wazzup.send_text(
                    channel_id=channel_id, chat_type=chat_type, chat_id=chat_id,
                    text=question,
                    crm_message_id=f"bot-q-{case['caseId']}-{message_id}"
                )
                await m.sessions.update_one(
                    {"channelId": channel_id, "chatId": chat_id},
                    {"$set": {"pendingQuestion": question, "updatedAt": datetime.utcnow()}},
                    upsert=True
                )
            continue

        # Ready to dispatch
        if case.get("caseType") == "lost_and_found":
            if settings.LOST_FOUND_TARGET and wazzup:
                await wazzup.send_text(
                    channel_id=channel_id,
                    chat_type=settings.LOST_FOUND_TARGET.get("chatType", "whatsgroup"),
                    chat_id=settings.LOST_FOUND_TARGET.get("chatId", chat_id),
                    text=format_dispatch_text(case),
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

        if case.get("caseType") == "complaint":
            ex = case.get("extracted", {})
            region = resolve_region(ex.get("train"), ex.get("routeFrom"), ex.get("routeTo"))
            executor = resolve_executor(region)

            if executor.target_chat_id and executor.target_chat_type and wazzup:
                await wazzup.send_text(
                    channel_id=channel_id,
                    chat_type=executor.target_chat_type,
                    chat_id=executor.target_chat_id,
                    text=format_dispatch_text(case) + f"\nРегион: {executor.region}",
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
            {"$set": {"currentStatus": st.get("status"), "statusRaw": st}},
            upsert=True
        )
