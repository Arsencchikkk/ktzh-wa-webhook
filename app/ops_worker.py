from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Dict

import httpx

from .db import MongoStore
from .settings import settings

log = logging.getLogger("ktzh")


def _fill_target_from_env(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Если target пустой — подставим дефолт из env.
    """
    tgt = dict(item.get("target") or {})
    tgt.setdefault("chatType", settings.OPS_CHAT_TYPE or "whatsapp")

    if not tgt.get("channelId"):
        tgt["channelId"] = settings.OPS_CHANNEL_ID or ""
    if not tgt.get("chatId"):
        tgt["chatId"] = settings.OPS_CHAT_ID or ""

    return tgt


async def _send_http(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = (settings.OPS_SEND_URL or "").strip()
    if not url:
        raise RuntimeError("OPS_SEND_URL is empty")

    headers = {}
    token = (settings.OPS_SEND_TOKEN or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeout = httpx.Timeout(15.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload, headers=headers)
        # лог полезен
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text[:500]}


async def run_worker(once: bool = False) -> None:
    store = MongoStore()
    await store.connect()
    if not store.enabled:
        log.warning("MongoStore disabled (no MONGODB_URI). Worker exits.")
        return

    log.info("OPS worker started. once=%s", once)

    try:
        while True:
            item = await store.claim_pending_outbox()
            if not item:
                if once:
                    return
                await asyncio.sleep(float(settings.OPS_POLL_SECONDS))
                continue

            outbox_id = item["_id"]
            attempts = int(item.get("attempts", 0))

            target = _fill_target_from_env(item)
            if not target.get("channelId") or not target.get("chatId"):
                err = "OPS target not configured (channelId/chatId empty)"
                log.warning("%s; outbox_id=%s", err, outbox_id)
                await store.mark_outbox_failed(outbox_id, err, attempts + 1)
                continue

            payload = {
                "channelId": target["channelId"],
                "chatId": target["chatId"],
                "chatType": target.get("chatType", "whatsapp"),
                "text": item.get("text", ""),
                "meta": {
                    "kind": item.get("kind"),
                    "caseId": item.get("caseId"),
                    "caseType": item.get("caseType"),
                    "source": item.get("source", {}),
                },
            }

            try:
                resp = await _send_http(payload)
                await store.mark_outbox_sent(outbox_id, resp=resp)
                log.info("OPS sent outbox_id=%s caseId=%s", outbox_id, item.get("caseId"))
            except Exception as e:
                err = str(e)
                log.warning("OPS send failed outbox_id=%s caseId=%s err=%s", outbox_id, item.get("caseId"), err)
                await store.mark_outbox_failed(outbox_id, err, attempts + 1)

            if once:
                return
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run single send attempt then exit (for cron)")
    args = parser.parse_args()
    asyncio.run(run_worker(once=args.once))


if __name__ == "__main__":
    main()
