from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from .settings import settings

log = logging.getLogger("ktzh")
router = APIRouter(prefix="/api/v1/ops", tags=["ops"])


class OpsSendIn(BaseModel):
    channelId: str
    chatId: str
    chatType: str = "whatsapp"
    text: str = Field(..., min_length=1)
    meta: Optional[Dict[str, Any]] = None


@router.post("/send")
async def ops_send(req: OpsSendIn, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """
    Принимает сообщение от воркера. Дальше здесь должен быть реальный "sender"
    (WhatsApp gateway / CRM API). Сейчас — минимальная заглушка + проверка токена.
    """

    token = (settings.OPS_SEND_TOKEN or "").strip()
    if token:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        if authorization.split(" ", 1)[1].strip() != token:
            raise HTTPException(status_code=403, detail="Invalid token")

    # TODO: тут должен быть реальный отправщик
    # await whatsapp_gateway_send(channelId=req.channelId, chatId=req.chatId, chatType=req.chatType, text=req.text)
    log.info("OPS SEND accepted -> channelId=%s chatId=%s type=%s text=%s",
             req.channelId, req.chatId, req.chatType, req.text[:200])

    return {"ok": True}
