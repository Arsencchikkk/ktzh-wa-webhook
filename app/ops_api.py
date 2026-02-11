from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .settings import settings

log = logging.getLogger("ktzh")
router = APIRouter(prefix="/api/v1/ops", tags=["ops"])


class OpsSendIn(BaseModel):
    channelId: str = Field(..., min_length=1)
    chatId: str = Field(..., min_length=1)
    chatType: str = "whatsapp"
    text: str = Field(..., min_length=1)
    meta: Optional[Dict[str, Any]] = None


def _check_bearer(authorization: Optional[str]) -> None:
    token = (settings.OPS_SEND_TOKEN or "").strip()
    if not token:
        # токен не задан — не защищаем эндпоинт (можно в проде всегда задавать)
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    got = authorization.split(" ", 1)[1].strip()
    if got != token:
        raise HTTPException(status_code=403, detail="Invalid token")


@router.post("/send")
async def ops_send(
    req: OpsSendIn,
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """
    Принимает сообщение от воркера и реально отправляет его в Wazzup (оперативникам как клиенту).
    """

    _check_bearer(authorization)

    # Берем WazzupClient из app.state (см. патч main.py ниже)
    wazzup = getattr(request.app.state, "wazzup", None)
    if wazzup is None:
        raise HTTPException(status_code=500, detail="WazzupClient is not initialized (app.state.wazzup missing)")

    try:
        resp = await wazzup.send_message(
            chat_id=req.chatId,
            channel_id=req.channelId,
            chat_type=req.chatType,
            text=req.text,
        )
    except Exception as e:
        log.exception(
            "OPS SEND: exception while sending to Wazzup -> channelId=%s chatId=%s type=%s",
            req.channelId, req.chatId, req.chatType
        )
        raise HTTPException(status_code=502, detail=f"Wazzup send exception: {e}")

    # WazzupClient у тебя возвращает {"ok": bool, ...}
    if not resp.get("ok"):
        log.warning(
            "OPS SEND: Wazzup returned ok=False -> channelId=%s chatId=%s resp=%s meta=%s",
            req.channelId, req.chatId, resp, (req.meta or {})
        )
        raise HTTPException(status_code=502, detail={"error": "wazzup_send_failed", "wazzup": resp})

    log.info(
        "OPS SEND: delivered via Wazzup -> channelId=%s chatId=%s type=%s msg=%s meta=%s",
        req.channelId, req.chatId, req.chatType,
        str(resp.get("response", {}))[:300],
        (req.meta or {})
    )

    return resp
