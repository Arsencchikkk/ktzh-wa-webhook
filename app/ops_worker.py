from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any, Dict, Optional

import httpx

from .db import MongoStore
from .settings import settings

log = logging.getLogger("ktzh.ops_worker")


def _s(val: Any) -> str:
    return "" if val is None else str(val)


def _get_cfg(name: str, default: Any = None) -> Any:
    # берем из settings если есть, иначе из env
    if hasattr(settings, name):
        v = getattr(settings, name)
        if v is not None and v != "":
            return v
    env = os.getenv(name)
    return env if env not in (None, "") else default


async def wazzup_send_message(
    client: httpx.AsyncClient,
    *,
    channel_id: str,
    chat_id: str,
    chat_type: str,
    text: str,
) -> Dict[str, Any]:
    base = _s(_get_cfg("WAZZUP_API_BASE", "https://api.wazzup24.com")).rstrip("/")
    token = _s(_get_cfg("WAZZUP_API_TOKEN", ""))

    if not token:
        raise RuntimeError("WAZZUP_API_TOKEN is empty")
    if not channel_id or not chat_id:
        raise RuntimeError("target channelId/chatId is empty")

    url = f"{base}/v3/message"
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "channelId": channel_id,
        "chatType": chat_type,
        "chatId": chat_id,
        "text": text,
    }

    r = await client.post(url, headers=headers, json=payload, timeout=20.0)
    data: Dict[str, Any]
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Wazzup HTTP {r.status_code}: {data}")

    return {"status_code": r.status_code, "data": data}


def _backoff_seconds(attempt: int) -> int:
    # 1,2,4,8,16,32... (max 10 min)
    sec = 2 ** max(0, attempt - 1)
    return min(sec, 600)


async def process_one(store: MongoStore, http: httpx.AsyncClient) -> bool:
    lock_seconds = int(_get_cfg("OPS_WORKER_LOCK_SECONDS", 60))
    max_retries = int(_get_cfg("OPS_WORKER_MAX_RETRIES", 6))

    doc = await store.claim_next_ops_message(lock_seconds=lock_seconds)
    if not doc:
        return False

    outbox_id = doc.get("_id")
    attempts = int(doc.get("attempts") or 0)

    try:
        target = doc.get("target") or {}
        channel_id = _s(target.get("channelId"))
        chat_id = _s(target.get("chatId"))
        chat_type = _s(target.get("chatType") or "whatsapp")
        text = _s(doc.get("text"))

        # защита от пустого текста
        if not text.strip():
            raise RuntimeError("outbox.text is empty")

        resp = await wazzup_send_message(
            http,
            channel_id=channel_id,
            chat_id=chat_id,
            chat_type=chat_type,
            text=text,
        )

        await store.mark_ops_sent(outbox_id, resp)
        log.info("SENT outbox=%s caseId=%s", outbox_id, doc.get("caseId"))
        return True

    except Exception as e:
        err = str(e)
        give_up = attempts >= max_retries
        retry_after = _backoff_seconds(attempts)

        await store.mark_ops_failed(outbox_id, err, retry_after_seconds=retry_after, give_up=give_up)

        if give_up:
            log.error("FAILED permanently outbox=%s caseId=%s err=%s", outbox_id, doc.get("caseId"), err)
        else:
            log.warning("FAILED retry outbox=%s caseId=%s in %ss err=%s", outbox_id, doc.get("caseId"), retry_after, err)

        return True  # обработали (даже если ошибка)


async def run_worker(once: bool = False) -> None:
    poll = int(_get_cfg("OPS_WORKER_POLL_SECONDS", 2))

    store = MongoStore()
    await store.connect()

    async with httpx.AsyncClient() as http:
        while True:
            did_any = False

            # выжимаем очередь до пустоты
            while True:
                ok = await process_one(store, http)
                if not ok:
                    break
                did_any = True

            if once:
                break

            if not did_any:
                await asyncio.sleep(poll)

    await store.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run one pass and exit (for cron)")
    args = parser.parse_args()

    asyncio.run(run_worker(once=args.once))


if __name__ == "__main__":
    main()
