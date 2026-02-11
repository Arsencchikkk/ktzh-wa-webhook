from __future__ import annotations

from typing import Any, Dict, Optional, List
import hashlib
import logging
import re

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .settings import settings
from .db import MongoStore
from .dialog import DialogManager, BotReply
from .wazzup_client import WazzupClient
from .ops_api import router as ops_router

log = logging.getLogger("ktzh")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.APP_NAME)
app.include_router(ops_router)

store = MongoStore()
dialog = DialogManager(store)
wazzup = WazzupClient(settings.WAZZUP_API_KEY)

# KTZH-20260211-94ED52-2761DB
CASE_ID_RE = re.compile(r"\bKTZH-\d{8}-[0-9A-F]{6}-[0-9A-F]{6}\b", re.I)


@app.on_event("startup")
async def startup():
    await store.connect()
    if getattr(store, "enabled", False):
        log.info("Mongo: ENABLED ✅")
    else:
        log.warning("Mongo: DISABLED ⚠️ (set MONGODB_URI / MONGO_URI in Render ENV)")


@app.on_event("shutdown")
async def shutdown():
    await store.close()
    if hasattr(wazzup, "close"):
        await wazzup.close()


def chat_hash(chat_id: str) -> str:
    raw = (settings.PHONE_HASH_SALT + "|" + (chat_id or "")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _check_token(request: Request) -> None:
    if settings.WEBHOOK_TOKEN:
        token = request.query_params.get("token") or request.query_params.get("crmKey")
        if token != settings.WEBHOOK_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid webhook token")


def _payload_to_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        out: List[Dict[str, Any]] = []
        for x in payload:
            out.extend(_payload_to_items(x))
        return out

    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("messages"), list):
        return [m for m in payload["messages"] if isinstance(m, dict)]

    # statuses — игнор
    if "statuses" in payload or isinstance(payload.get("statuses"), list):
        return []

    return [payload]


def extract_inbound(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # wrappers сюда не должны попадать
    if "statuses" in payload or "messages" in payload:
        return None

    # echo нашего бота — игнор
    if payload.get("isEcho") is True:
        return None

    direction = payload.get("direction")
    if direction and direction != "inbound":
        return None

    text = payload.get("text")
    if not text:
        raw = payload.get("raw") or {}
        if isinstance(raw, dict):
            text = raw.get("text")

    if not text:
        content = payload.get("content") or {}
        if isinstance(content, dict):
            text = content.get("text")

    if not text:
        return None

    return {
        "chatId": str(payload.get("chatId") or ""),
        "channelId": str(payload.get("channelId") or ""),
        "chatType": str(payload.get("chatType") or "whatsapp"),
        "dateTime": payload.get("dateTime"),
        "text": str(text),
        "raw": payload,
    }


def _is_ops_chat(msg: Dict[str, Any]) -> bool:
    ops_channel = (settings.OPS_CHANNEL_ID or "").strip()
    ops_chat = (settings.OPS_CHAT_ID or "").strip()
    if not ops_channel or not ops_chat:
        return False
    return (msg.get("channelId") == ops_channel) and (msg.get("chatId") == ops_chat)


def _extract_case_id(text: str) -> Optional[str]:
    m = CASE_ID_RE.search(text or "")
    return m.group(0).upper() if m else None


def _looks_resolved(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False

    # ключевые слова “закрытия”
    keywords = (
        "закрыт", "закрыли", "закрыта", "закрыто",
        "обработан", "обработано", "обработали",
        "решено", "решили", "выполнено", "готово",
        "вернули", "возвратили", "передали владельцу", "владельцу",
        "нашли", "найдено",
        "resolved", "closed", "done", "completed",
    )
    return any(k in t for k in keywords)


async def _handle_ops_incoming(msg: Dict[str, Any]) -> None:
    """
    OPS пишет в WhatsApp чат -> сохраняем в messages как ops_in.
    Если текст содержит caseId и выглядит как “решено/закрыто” -> закрываем кейс в cases.
    """
    text = msg.get("text") or ""
    case_id = _extract_case_id(text)

    log.info("OPS IN: text=%r caseId=%s", text, case_id)

    # 1) логируем входящее от OPS
    if hasattr(store, "add_message"):
        try:
            await store.add_message({
                "dir": "ops_in",
                "chatIdHash": "",
                "chatId": msg.get("chatId"),
                "channelId": msg.get("channelId"),
                "chatType": msg.get("chatType"),
                "text": text,
                "raw": msg.get("raw"),
                "meta": {"caseId": case_id} if case_id else {},
            })
        except Exception:
            log.warning("Mongo add_message failed for ops_in", exc_info=True)

    # 2) если есть caseId и текст похож на “решено” -> закрываем кейс
    if case_id and _looks_resolved(text) and hasattr(store, "close_case"):
        try:
            closed = await store.close_case(
                case_id=case_id,
                resolution_text=text,
                meta={
                    "channelId": msg.get("channelId"),
                    "chatId": msg.get("chatId"),
                    "chatType": msg.get("chatType"),
                },
            )
            if closed:
                log.info("CASE CLOSED ✅ caseId=%s", case_id)
            else:
                log.warning("CASE NOT FOUND or already closed: %s", case_id)
        except Exception:
            log.warning("close_case failed", exc_info=True)


async def _send_to_ops_if_needed(reply: BotReply) -> None:
    """
    Если dialog вернул meta.ops -> отправляем текст в OPS чат.
    """
    ops = (reply.meta or {}).get("ops")
    if not isinstance(ops, dict):
        return

    text = str(ops.get("text") or "").strip()
    if not text:
        return

    ops_channel = (settings.OPS_CHANNEL_ID or "").strip()
    ops_chat = (settings.OPS_CHAT_ID or "").strip()
    ops_type = (settings.OPS_CHAT_TYPE or "whatsapp").strip()

    if not ops_channel or not ops_chat:
        log.warning("OPS target not configured: OPS_CHANNEL_ID/OPS_CHAT_ID empty")
        return

    res = await wazzup.send_message(
        chat_id=ops_chat,
        channel_id=ops_channel,
        chat_type=ops_type,
        text=text,
    )
    log.info("OPS SENT: %s", res)

    # лог в mongo
    if hasattr(store, "add_message"):
        try:
            await store.add_message({
                "dir": "ops_out",
                "chatIdHash": "",
                "chatId": ops_chat,
                "channelId": ops_channel,
                "chatType": ops_type,
                "text": text,
                "send": res,
                "meta": {"caseId": ops.get("caseId"), "caseType": ops.get("caseType")},
            })
        except Exception:
            log.warning("Mongo add_message failed for ops_out", exc_info=True)


async def process_items(items: List[Dict[str, Any]]) -> None:
    log.info("WEBHOOK: got %s item(s)", len(items))

    for item in items:
        msg = extract_inbound(item)

        if not msg or not msg["chatId"]:
            log.info("SKIP: not inbound text payload keys=%s", list(item.keys())[:20])
            continue

        # ✅ 0) если это OPS чат — НЕ гоним через dialog, а обрабатываем закрытие и выходим
        if _is_ops_chat(msg):
            await _handle_ops_incoming(msg)
            continue

        # ✅ 1) TEST MODE только для клиентов (OPS не трогаем)
        if settings.TEST_MODE:
            allowed_chat = (settings.TEST_CHAT_ID or "").strip()
            allowed_channel = (settings.TEST_CHANNEL_ID or "").strip()

            if not allowed_chat:
                log.warning("TEST_MODE enabled but TEST_CHAT_ID empty")
                continue

            if msg["chatId"] != allowed_chat:
                log.info("TEST_MODE: SKIP chatId=%s", msg["chatId"])
                continue

            if allowed_channel and msg["channelId"] != allowed_channel:
                log.info("TEST_MODE: SKIP channelId=%s", msg["channelId"])
                continue

        chat_id_hash = chat_hash(msg["chatId"])
        log.info("IN: chatId=%s text=%r", msg["chatId"], msg["text"])

        # входящее в mongo
        if hasattr(store, "add_message"):
            await store.add_message({
                "dir": "in",
                "chatIdHash": chat_id_hash,
                "chatId": msg["chatId"],
                "channelId": msg["channelId"],
                "chatType": msg["chatType"],
                "text": msg["text"],
                "raw": msg["raw"],
            })

        try:
            bot_reply = await dialog.handle(
                chat_id_hash=chat_id_hash,
                chat_meta={
                    "chatId": msg["chatId"],
                    "channelId": msg["channelId"],
                    "chatType": msg["chatType"],
                    "dateTime": msg["dateTime"],
                    "raw": msg["raw"],
                },
                user_text=msg["text"],
            )
        except Exception as e:
            log.exception("Dialog error: %s", e)
            bot_reply = BotReply(text="Извините, произошла ошибка. Попробуйте ещё раз.")

        log.info("BOT: reply=%r", bot_reply.text)

        # отправка клиенту
        if settings.BOT_SEND_ENABLED:
            send_res = await wazzup.send_message(
                chat_id=msg["chatId"],
                channel_id=msg["channelId"],
                chat_type=msg["chatType"],
                text=bot_reply.text,
            )
            log.info("SENT: %s", send_res)

            if hasattr(store, "add_message"):
                await store.add_message({
                    "dir": "out",
                    "chatIdHash": chat_id_hash,
                    "chatId": msg["chatId"],
                    "channelId": msg["channelId"],
                    "chatType": msg["chatType"],
                    "text": bot_reply.text,
                    "send": send_res,
                })

        # ✅ отправка оперативникам (если dialog вернул meta.ops)
        try:
            await _send_to_ops_if_needed(bot_reply)
        except Exception:
            log.warning("OPS send failed", exc_info=True)


@app.get("/")
def root():
    return {"ok": True, "service": settings.APP_NAME}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/webhooks")
async def webhooks_alias(request: Request, background: BackgroundTasks):
    return await wazzup_webhook(request, background)


@app.post("/webhook/wazzup")
async def wazzup_webhook(request: Request, background: BackgroundTasks):
    _check_token(request)

    payload = await request.json()
    items = _payload_to_items(payload)

    if not items:
        return JSONResponse({"ok": True, "ignored": True})

    background.add_task(process_items, items)
    return JSONResponse({"ok": True, "queued": len(items)})
