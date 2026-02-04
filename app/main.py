import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from dateutil.parser import isoparse

from .db import connect_mongo, get_collection

load_dotenv()

WAZZUP_CRM_KEY = os.getenv("WAZZUP_CRM_KEY", "").strip()

app = FastAPI(title="Wazzup Webhook Receiver", version="1.0.0")


@app.on_event("startup")
async def startup():
    await connect_mongo()


@app.get("/")
async def health():
    return {"ok": True, "service": "ktzh-wazzup-webhook"}


def _auth_ok(req: Request) -> bool:
    """Если WAZZUP_CRM_KEY задан — требуем Authorization: Bearer <crmKey>.
    Если ключ пуст — не проверяем (разрешаем без заголовка).
    """
    if not WAZZUP_CRM_KEY:
        return True
    auth = req.headers.get("authorization", "")
    return auth == f"Bearer {WAZZUP_CRM_KEY}"


def _safe_parse_dt(dt_str: Optional[str]):
    if not dt_str:
        return None
    try:
        return isoparse(dt_str)
    except Exception:
        return None


async def _process_messages_and_statuses(payload: Dict[str, Any]):
    col = get_collection()

    # TEST webhook при подписке
    if payload.get("test") is True:
        return

    messages: List[Dict[str, Any]] = payload.get("messages") or []
    statuses: List[Dict[str, Any]] = payload.get("statuses") or []

    # 1) messages
    for m in messages:
        message_id = m.get("messageId")
        if not message_id:
            continue

        is_echo = bool(m.get("isEcho", False))
        direction = "outbound" if is_echo else "inbound"

        dt = _safe_parse_dt(m.get("dateTime"))
        msg_type = m.get("type")
        text = m.get("text")

        # Нормализованный документ (raw всегда сохраняем)
        doc = {
            "messageId": message_id,
            "channelId": m.get("channelId"),
            "chatType": m.get("chatType"),
            "chatId": m.get("chatId"),
            "dateTime": dt,
            "type": msg_type,
            "direction": direction,
            "isEcho": is_echo,

            "contact": m.get("contact"),
            "authorName": m.get("authorName"),
            "authorId": m.get("authorId"),

            "text": text,
            "contentUri": m.get("contentUri"),

            "isEdited": bool(m.get("isEdited", False)),
            "isDeleted": bool(m.get("isDeleted", False)),
            "oldInfo": m.get("oldInfo"),

            # Wazzup иногда в messages кладёт status (например inbound)
            "currentStatus": m.get("status") or ("inbound" if not is_echo else None),

            "raw": m,
        }

        # upsert по messageId
        await col.update_one(
            {"messageId": message_id},
            {
                "$setOnInsert": doc,
                "$set": {
                    # обновляем то, что могло измениться при edit/delete
                    "text": text,
                    "contentUri": m.get("contentUri"),
                    "isEdited": bool(m.get("isEdited", False)),
                    "isDeleted": bool(m.get("isDeleted", False)),
                    "oldInfo": m.get("oldInfo"),
                    "raw": m,
                    "currentStatus": doc.get("currentStatus"),
                },
            },
            upsert=True,
        )

    # 2) statuses (статусы исходящих)
    for s in statuses:
        message_id = s.get("messageId")
        if not message_id:
            continue

        dt = _safe_parse_dt(s.get("dateTime"))
        status = s.get("status")

        status_event = {
            "status": status,
            "dateTime": dt,
            "raw": s,
        }

        # если сообщения ещё нет — создадим "stub"
        await col.update_one(
            {"messageId": message_id},
            {
                "$setOnInsert": {
                    "messageId": message_id,
                    "channelId": s.get("channelId"),
                    "chatType": s.get("chatType"),
                    "chatId": s.get("chatId"),
                    "type": s.get("type"),
                    "direction": "outbound",  # статусы обычно для исходящих
                    "raw": {"createdFrom": "statusWebhook"},
                },
                "$set": {
                    "currentStatus": status,
                    "statusUpdatedAt": dt,
                },
                "$push": {
                    "statusHistory": status_event
                },
            },
            upsert=True,
        )


@app.post("/webhooks")
async def wazzup_webhooks(request: Request, background: BackgroundTasks):
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"ok": True}, status_code=200)
    except Exception:
        # Если тело не JSON — всё равно 200, чтобы не было бесконечных ретраев
        return JSONResponse({"ok": True}, status_code=200)

    # Быстрый 200 OK, обработка в фоне
    background.add_task(_process_messages_and_statuses, payload)

    return JSONResponse({"ok": True}, status_code=200)
