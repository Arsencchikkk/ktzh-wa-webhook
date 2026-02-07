from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

from .nlu import run_nlu
from .routing import resolve_region, resolve_executor
from . import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)



def _to_aware_utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        # naive -> assume UTC
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _compact(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None and v != "" and v != [] and v != {}}


def _case_id() -> str:
    return f"KTZH-{_now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"


def _is_stale(ts: Optional[datetime], hours: int = 24) -> bool:
    if not ts:
        return True
    # если вдруг в базе оказалось naive время — считаем UTC
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (_now() - ts) > timedelta(hours=hours)



def required_slots(case: Dict[str, Any]) -> List[str]:
    ctype = case.get("caseType")
    extracted = case.get("extracted", {})
    cats = set(case.get("categories", []))

    if ctype == "complaint":
        need = []
        if not extracted.get("train") and not (extracted.get("routeFrom") and extracted.get("routeTo")):
            need.append("train_or_route")

        wagon_needed = bool(cats.intersection({"температура", "санитария", "сервис", "проводник"}))
        if wagon_needed and not extracted.get("wagon"):
            need.append("wagon")

        if "опоздание" in cats:
            if not extracted.get("date") and not extracted.get("time"):
                need.append("date_or_time")

        return need

    if ctype == "lost_and_found":
        need = []
        if not extracted.get("item"):
            need.append("item")
        if not extracted.get("train") and not (extracted.get("routeFrom") and extracted.get("routeTo")):
            need.append("train_or_route")
        if not extracted.get("wagon") and not extracted.get("date") and not extracted.get("time"):
            need.append("wagon_or_time")
        return need

    return []


def build_question(case: Dict[str, Any], missing: List[str]) -> Optional[str]:
    if not missing:
        return None

    if missing[0] == "train_or_route":
        if "wagon" in missing:
            return "Уточните, пожалуйста, номер поезда и вагон (пример: Т58, вагон 3)."
        return "Уточните, пожалуйста, номер поезда (например Т58) или маршрут (например Семей–Кызылорда)."

    if missing[0] == "wagon":
        return "Уточните, пожалуйста, номер вагона."

    if missing[0] == "date_or_time":
        return "Уточните, пожалуйста, дату или время поездки (как минимум одно)."

    if missing[0] == "item":
        return "Подскажите, пожалуйста, что именно вы забыли/потеряли (предмет, цвет, особые приметы)."

    if missing[0] == "wagon_or_time":
        return "Уточните, пожалуйста, номер вагона или время/дату, когда оставили вещь (как минимум одно)."

    return "Уточните, пожалуйста, детали обращения."


def format_dispatch_text(case: Dict[str, Any]) -> str:
    ex = case.get("extracted", {})
    cats = ", ".join(case.get("categories", [])) or "не определено"
    sev = case.get("severity", {}).get("score", 1)

    parts = [
        f"Новый кейс {case.get('caseId')}",
        f"Тип: {case.get('caseType')}",
        f"Категории: {cats}",
        f"Серьёзность: {sev}/5",
        f"Клиент: {case.get('chatId')}",
    ]

    if ex.get("train"):
        parts.append(f"Поезд: {ex.get('train')}")
    if ex.get("wagon"):
        parts.append(f"Вагон: {ex.get('wagon')}")
    if ex.get("routeFrom") and ex.get("routeTo"):
        parts.append(f"Маршрут: {ex.get('routeFrom')} – {ex.get('routeTo')}")
    if ex.get("date") or ex.get("time"):
        parts.append(f"Дата/время: {ex.get('date','')} {ex.get('time','')}".strip())

    if ex.get("item"):
        parts.append(f"Детали: {ex.get('item')}")
    else:
        parts.append(f"Текст: {case.get('lastText','')}".strip())

    att = case.get("attachments", [])
    if att:
        parts.append("Вложения:")
        for a in att[:5]:
            parts.append(f"- {a.get('type')}: {a.get('contentUri') or a.get('text') or ''}".strip())

    return "\n".join([p for p in parts if p])


def format_user_ack(case: Dict[str, Any]) -> str:
    if case.get("caseType") == "gratitude":
        return "Спасибо за обратную связь! Передадим благодарность команде 🙏"
    if case.get("caseType") == "lost_and_found":
        return f"Принял(а). Передал(а) информацию по забытым вещам. Номер заявки: {case.get('caseId')}."
    if case.get("caseType") == "complaint":
        return f"Принял(а) обращение. Номер заявки: {case.get('caseId')}. Передал(а) ответственным."
    if case.get("caseType") == "info":
        return f"Ваш вопрос принят. Передал(а) оператору. Номер: {case.get('caseId')}."
    return "Принял(а)."


async def ensure_session(mongo, channel_id: str, chat_id: str, chat_type: str) -> Dict[str, Any]:
    sess = await mongo.sessions.find_one({"channelId": channel_id, "chatId": chat_id})
    if not sess:
        sess = {
            "channelId": channel_id,
            "chatId": chat_id,
            "chatType": chat_type,
            "activeCaseId": None,
            "pendingQuestion": None,
            "createdAt": _now(),
            "updatedAt": _now(),
        }
        await mongo.sessions.insert_one(sess)
        sess = await mongo.sessions.find_one({"channelId": channel_id, "chatId": chat_id})
    return sess


async def load_active_case(mongo, sess: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cid = sess.get("activeCaseId")
    if not cid:
        return None
    case = await mongo.cases.find_one({"caseId": cid})
    if not case:
        return None
    if case.get("status") in ("sent", "closed"):
        return None
    if _is_stale(case.get("updatedAt"), hours=24):
        return None
    return case


async def create_case(
    mongo,
    *,
    channel_id: str,
    chat_id: str,
    chat_type: str,
    contact_name: Optional[str],
    case_type: str,
    nlu
) -> Dict[str, Any]:
    cid = _case_id()
    case = {
        "caseId": cid,
        "status": "collecting",
        "caseType": case_type,
        "channelId": channel_id,
        "chatId": chat_id,
        "chatType": chat_type,
        "contactName": contact_name,
        "language": nlu.language,
        "categories": nlu.categories,
        "severity": nlu.severity,
        "extracted": _compact(nlu.slots),
        "evidence": [],
        "attachments": [],
        "lastText": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    await mongo.cases.insert_one(case)
    await mongo.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {"activeCaseId": cid, "updatedAt": _now(), "pendingQuestion": None}},
        upsert=True,
    )
    return await mongo.cases.find_one({"caseId": cid})


async def update_case_with_message(mongo, case: Dict[str, Any], message: Dict[str, Any], nlu) -> Dict[str, Any]:
    ex = case.get("extracted", {})
    for k, v in nlu.slots.items():
        if v is not None:
            ex[k] = v

    cats = list(dict.fromkeys((case.get("categories") or []) + (nlu.categories or [])))

    ev = {
        "messageId": message.get("messageId"),
        "dateTime": message.get("dateTime"),
        "type": message.get("type"),
        "text": message.get("text"),
        "contentUri": message.get("contentUri"),
    }

    upd = {
        "$set": {
            "updatedAt": _now(),
            "language": nlu.language,
            "categories": cats,
            "severity": case.get("severity") if case.get("caseType") != "complaint" else nlu.severity,
            "extracted": _compact(ex),
            "lastText": message.get("text") or case.get("lastText"),
        },
        "$push": {"evidence": ev},
    }

    if message.get("contentUri") and message.get("type") != "text":
        upd["$push"]["attachments"] = {
            "type": message.get("type"),
            "text": message.get("text"),
            "contentUri": message.get("contentUri"),
            "messageId": message.get("messageId"),
        }

    await mongo.cases.update_one({"caseId": case["caseId"]}, upd)
    return await mongo.cases.find_one({"caseId": case["caseId"]})


async def close_case(mongo, case_id: str, status: str = "closed"):
    await mongo.cases.update_one({"caseId": case_id}, {"$set": {"status": status, "updatedAt": _now()}})
