from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import settings


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _gen_case_id() -> str:
    # KTZH-YYYYMMDD-XXXXXXXX
    d = _now_utc().strftime("%Y%m%d")
    tail = secrets.token_hex(4).upper()
    return f"KTZH-{d}-{tail}"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _extract_train_from_text(text: str) -> Optional[str]:
    t = (text or "").strip()
    m = re.search(r"\b([TТ])\s*[-]?\s*(\d{1,4}[A-Za-zА-Яа-яЁё]?)\b", t)
    if not m:
        return None
    return (m.group(1) + m.group(2)).upper().replace(" ", "").replace("-", "")


def _extract_wagon_from_text(text: str) -> Optional[int]:
    t = _norm(text)
    m = re.search(r"\bвагон\s*(\d{1,2})\b", t)
    if m:
        return int(m.group(1))
    if t.isdigit():
        return int(t)
    return None


def _extract_seat_from_text(text: str) -> Optional[int]:
    t = _norm(text)
    m = re.search(r"\bместо\s*(\d{1,3})\b", t)
    if m:
        return int(m.group(1))
    return None


def _extract_item_guess(text: str) -> Optional[str]:
    """
    Very simple heuristic: if message contains lost keywords, keep whole message as item description.
    """
    t = _norm(text)
    if any(k in t for k in ["забыл", "потерял", "оставил", "ұмыт", "жоғалт", "forgot", "lost"]):
        return text.strip()
    return None


def _slot_key_aliases() -> Dict[str, List[str]]:
    return {
        "train": ["train", "poezd", "поезд", "т", "t"],
        "wagon": ["wagon", "car", "вагон"],
        "seat": ["seat", "place", "место"],
        "routeFrom": ["from", "routeFrom", "откуда", "станция_отправления"],
        "routeTo": ["to", "routeTo", "куда", "станция_назначения"],
        "item": ["item", "lostItem", "thing", "вещь", "предмет"],
        "details": ["details", "desc", "description", "problem", "жалоба", "текст"],
        "gratitudeText": ["gratitudeText", "thanks", "благодарность_текст"],
    }


def _merge_extracted(extracted: Dict[str, Any], nlu: Any, message_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge NLU slots + heuristic extraction from text.
    """
    text = (message_doc.get("text") or "").strip()

    slots = {}
    try:
        slots = getattr(nlu, "slots", {}) or {}
    except Exception:
        slots = {}

    # normalize slot names
    aliases = _slot_key_aliases()

    # copy known aliases from slots into extracted
    for target_key, keys in aliases.items():
        for k in keys:
            if isinstance(slots, dict) and k in slots and slots[k] not in (None, "", []):
                extracted.setdefault(target_key, slots[k])

    # heuristics from text if missing
    if not extracted.get("train"):
        tr = _extract_train_from_text(text)
        if tr:
            extracted["train"] = tr

    if extracted.get("wagon") in (None, "", 0):
        w = _extract_wagon_from_text(text)
        if w is not None:
            extracted["wagon"] = w

    if not extracted.get("seat"):
        s = _extract_seat_from_text(text)
        if s is not None:
            extracted["seat"] = s

    # complaint details
    if not extracted.get("details") and text:
        extracted["details"] = text

    # item guess for lost&found
    if not extracted.get("item"):
        it = _extract_item_guess(text)
        if it:
            extracted["item"] = it

    # gratitude details: if message is not only "спасибо" etc.
    if text and len(text) >= 10 and any(k in _norm(text) for k in ["спасибо", "благодар", "рахмет", "алғыс"]):
        extracted.setdefault("gratitudeText", text)

    return extracted


async def ensure_session(m, channel_id: str, chat_id: str, chat_type: str) -> Dict[str, Any]:
    now = _now_utc()
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {
            "$setOnInsert": {
                "channelId": channel_id,
                "chatId": chat_id,
                "createdAt": now,
                "draftSlots": {},
            },
            "$set": {
                "chatType": chat_type,
                "updatedAt": now,
            },
        },
        upsert=True,
    )
    sess = await m.sessions.find_one({"channelId": channel_id, "chatId": chat_id})
    return sess or {}



async def load_active_case(m, session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cid = (session or {}).get("activeCaseId")
    if not cid:
        return None
    case = await m.cases.find_one({"caseId": cid})
    if not case:
        return None
    # if already closed/sent -> do not continue it
    if case.get("status") in ("closed", "sent"):
        return None
    return case


async def create_case(
    m,
    channel_id: str,
    chat_id: str,
    chat_type: str,
    contact_name: Optional[str],
    case_type: str,
    nlu: Any,
) -> Dict[str, Any]:
    now = _now_utc()
    case_id = _gen_case_id()

    language = getattr(nlu, "language", None) or "ru"
    case_type = case_type or "other"

    doc = {
        "caseId": case_id,
        "status": "collecting",
        "caseType": case_type,
        "channelId": channel_id,
        "chatId": chat_id,
        "chatType": chat_type,
        "contactName": contact_name,
        "language": language,
        "categories": [],
        "severity": {},
        "extracted": {},
        "evidence": [],
        "attachments": [],
        "lastText": None,
        "createdAt": now,
        "updatedAt": now,
    }

    await m.cases.insert_one(doc)

    # bind session
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {"activeCaseId": case_id, "pendingSlot": None, "pendingQuestion": None, "updatedAt": now}},
        upsert=True,
    )

    return await m.cases.find_one({"caseId": case_id})


async def _apply_pending_slot_answer(m, case: Dict[str, Any], message_doc: Dict[str, Any]) -> None:
    """
    If bot previously asked something, interpret current message as answer.
    """
    channel_id = case["channelId"]
    chat_id = case["chatId"]

    sess = await m.sessions.find_one({"channelId": channel_id, "chatId": chat_id})
    pending_slot = (sess or {}).get("pendingSlot")
    if not pending_slot:
        return

    text = (message_doc.get("text") or "").strip()
    ex = case.get("extracted", {}) or {}

    # map slot -> how to parse
    if pending_slot == "train":
        tr = _extract_train_from_text(text)
        if tr:
            ex["train"] = tr

    elif pending_slot == "wagon":
        w = _extract_wagon_from_text(text)
        if w is not None:
            ex["wagon"] = w

    elif pending_slot == "details":
        if text:
            ex["details"] = text

    elif pending_slot == "gratitudeText":
        if text and len(text) >= 5:
            ex["gratitudeText"] = text

    elif pending_slot == "item":
        if text and len(text) >= 3:
            ex["item"] = text

    elif pending_slot == "seat":
        s = _extract_seat_from_text(text)
        if s is not None:
            ex["seat"] = s
        else:
            # allow "12" as seat
            if _norm(text).isdigit():
                ex["seat"] = int(_norm(text))

    # clear pending slot
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {"pendingSlot": None, "pendingQuestion": None, "updatedAt": _now_utc()}},
        upsert=True,
    )

    await m.cases.update_one(
        {"caseId": case["caseId"]},
        {"$set": {"extracted": ex, "updatedAt": _now_utc()}},
    )


async def update_case_with_message(m, case: Dict[str, Any], message_doc: Dict[str, Any], nlu: Any) -> Dict[str, Any]:
    """
    - append evidence
    - merge extracted fields from NLU + heuristics
    - apply pendingSlot answer if any
    """
    now = _now_utc()

    ev = {
        "messageId": message_doc.get("messageId"),
        "dateTime": message_doc.get("dateTime"),
        "text": message_doc.get("text"),
        "contentUri": message_doc.get("contentUri"),
        "type": message_doc.get("type"),
        "direction": message_doc.get("direction"),
    }

    extracted = case.get("extracted", {}) or {}
    extracted = _merge_extracted(extracted, nlu, message_doc)

    # attachments list
    attachments = case.get("attachments", []) or []
    if message_doc.get("contentUri"):
        attachments.append(
            {
                "messageId": message_doc.get("messageId"),
                "contentUri": message_doc.get("contentUri"),
                "type": message_doc.get("type"),
                "dateTime": message_doc.get("dateTime"),
            }
        )

    await m.cases.update_one(
        {"caseId": case["caseId"]},
        {
            "$set": {
                "lastText": (message_doc.get("text") or "").strip(),
                "extracted": extracted,
                "attachments": attachments,
                "updatedAt": now,
            },
            "$push": {"evidence": ev},
        },
    )

    # reload and apply pending slot answer (so "4" works correctly)
    fresh = await m.cases.find_one({"caseId": case["caseId"]})
    await _apply_pending_slot_answer(m, fresh, message_doc)

    return await m.cases.find_one({"caseId": case["caseId"]})


def required_slots(case: Dict[str, Any]) -> List[str]:
    """
    Minimal "human" requirements:
    - complaint: train + wagon + details (details can be initial message)
    - gratitude: gratitudeText (not just "Благодарность")
    - lost_and_found: train + item + seat (wagon optional but desirable)
    - info: details
    """
    ex = case.get("extracted", {}) or {}
    ctype = case.get("caseType") or "other"

    missing: List[str] = []

    if ctype == "complaint":
        if not ex.get("train"):
            missing.append("train")
        if ex.get("wagon") in (None, "", 0):
            missing.append("wagon")
        # details must be meaningful
        details = (ex.get("details") or "").strip()
        if not details or len(details) < 8:
            missing.append("details")

    elif ctype == "gratitude":
        gt = (ex.get("gratitudeText") or "").strip()
        # if only "благодарность/спасибо" => ask more
        if not gt or len(gt) < 10 or _norm(gt) in {"благодарность", "спасибо", "рахмет", "алғыс"}:
            missing.append("gratitudeText")

    elif ctype == "lost_and_found":
        if not ex.get("train"):
            missing.append("train")
        item = (ex.get("item") or "").strip()
        if not item or len(item) < 5:
            missing.append("item")
        if not ex.get("seat"):
            missing.append("seat")

    elif ctype == "info":
        details = (ex.get("details") or "").strip()
        if not details or len(details) < 5:
            missing.append("details")

    return missing


def build_question(case: Dict[str, Any], missing: List[str]) -> str:
    lang = case.get("language") or "ru"
    slot = missing[0] if missing else None

    # RU/KZ minimal
    ru = {
        "train": "Уточните, пожалуйста, номер поезда (например: Т58).",
        "wagon": "Уточните, пожалуйста, номер вагона.",
        "details": "Опишите, пожалуйста, подробнее, что именно случилось (1–2 предложения).",
        "gratitudeText": "Спасибо! Напишите, пожалуйста, за что именно хотите поблагодарить (пару предложений).",
        "item": "Что именно вы забыли? Опишите предмет (цвет/бренд/что внутри).",
        "seat": "Укажите ваше место (номер места) или хотя бы расположение (верх/низ, купе/плацкарт).",
    }
    kk = {
        "train": "Пойыз нөмірін жазыңыз (мысалы: Т58).",
        "wagon": "Вагон нөмірін жазыңыз.",
        "details": "Не болғанын қысқаша жазыңыз (1–2 сөйлем).",
        "gratitudeText": "Рақмет! Кімге/не үшін алғыс айтқыңыз келеді? Қысқаша жазыңыз.",
        "item": "Нені ұмытып кеттіңіз? Затты сипаттаңыз (түсі/бренд/ішінде не бар).",
        "seat": "Орын нөмірін жазыңыз немесе орналасуын (жоғары/төмен, купе/плацкарт).",
    }

    table = kk if lang == "kk" else ru
    return table.get(slot, "Уточните, пожалуйста, детали.")


def format_dispatch_text(case: Dict[str, Any]) -> str:
    ex = case.get("extracted", {}) or {}
    ev = case.get("evidence", []) or []
    last_msgs = []
    for x in ev[-3:]:
        t = (x.get("text") or "").strip()
        if t:
            last_msgs.append(t)

    train = ex.get("train")
    wagon = ex.get("wagon")
    seat = ex.get("seat")
    item = ex.get("item")
    details = ex.get("details")
    gt = ex.get("gratitudeText")

    lines = [
        f"Заявка: {case.get('caseId')}",
        f"Тип: {case.get('caseType')}",
        f"Контакт: {case.get('contactName') or '-'}",
        f"Чат: {case.get('chatId')} ({case.get('chatType')})",
    ]

    if train:
        lines.append(f"Поезд: {train}")
    if wagon:
        lines.append(f"Вагон: {wagon}")
    if seat:
        lines.append(f"Место: {seat}")

    if case.get("caseType") == "lost_and_found":
        if item:
            lines.append(f"Что забыли: {item}")

    if case.get("caseType") == "gratitude":
        if gt:
            lines.append(f"Текст благодарности: {gt}")

    if case.get("caseType") in ("complaint", "info"):
        if details:
            lines.append(f"Описание: {details}")

    if last_msgs:
        lines.append("Последние сообщения:")
        for t in last_msgs:
            lines.append(f"- {t}")

    if case.get("attachments"):
        lines.append(f"Вложения: {len(case.get('attachments') or [])}")

    return "\n".join(lines)


def format_user_ack(case: Dict[str, Any]) -> str:
    ctype = case.get("caseType")
    cid = case.get("caseId")

    if ctype == "complaint":
        return f"Принял(а) вашу жалобу. Номер заявки: {cid}. Передаю ответственным, спасибо."
    if ctype == "lost_and_found":
        return f"Принял(а) информацию по забытым вещам. Номер заявки: {cid}. Мы передали в службу найденных вещей."
    if ctype == "info":
        return f"Спасибо! Ваш запрос принят. Номер: {cid}. Мы передали оператору."
    if ctype == "gratitude":
        return f"Спасибо! Номер: {cid}. Передадим вашу благодарность 🙏"
    return f"Спасибо! Номер: {cid}."


async def close_case(m, case_id: str, status: str = "closed") -> None:
    await m.cases.update_one(
        {"caseId": case_id},
        {"$set": {"status": status, "updatedAt": _now_utc()}},
    )
