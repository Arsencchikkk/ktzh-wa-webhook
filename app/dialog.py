from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# time helpers (tz-aware)
# ----------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------
# regex / extraction
# ----------------------------
TRAIN_RE = re.compile(r"\b[тt]\s*-?\s*(\d{1,4})\b", re.IGNORECASE)

# "вагон 8" и "8 вагон"
CAR_RE_1 = re.compile(r"\bвагон\s*(\d{1,2})\b", re.IGNORECASE)
CAR_RE_2 = re.compile(r"\b(\d{1,2})\s*вагон\b", re.IGNORECASE)

SEAT_RE_1 = re.compile(r"\bместо\s*(\d{1,3}[а-яa-z]?)\b", re.IGNORECASE)
SEAT_RE_2 = re.compile(r"\b(\d{1,3}[а-яa-z]?)\s*место\b", re.IGNORECASE)

# staff for gratitude
STAFF_RE = re.compile(
    r"\b(проводник|кондуктор|кассир|стюард|начальник\s+поезда)\b\s*([А-ЯA-ZЁӘІҢҒҮҰҚӨҺ][а-яa-zёәіңғүұқөһ-]{1,40})?",
    re.IGNORECASE
)

# lost item hint
ITEM_AFTER_VERB_RE = re.compile(r"\b(забыл|оставил|потерял|утерял)\b\s+(.+)$", re.IGNORECASE)


def _first_int(text: str) -> Optional[int]:
    m = re.search(r"(\d{1,4})", text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def extract_entities(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    out: Dict[str, Any] = {}

    mt = TRAIN_RE.search(t)
    if mt:
        out["train"] = f"T{mt.group(1)}".upper()

    mc = CAR_RE_1.search(t) or CAR_RE_2.search(t)
    if mc:
        out["carNumber"] = int(mc.group(1))

    ms = SEAT_RE_1.search(t) or SEAT_RE_2.search(t)
    if ms:
        out["seat"] = ms.group(1).upper()

    msf = STAFF_RE.search(t)
    if msf:
        out["staffRole"] = msf.group(1).lower()
        if msf.group(2):
            out["staffName"] = msf.group(2).strip()

    return out


# ----------------------------
# Session
# ----------------------------
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
                "activeCases": {},          # {"complaint": caseId, "lost_and_found": caseId, ...}
                "activeCaseId": None,       # optional primary
                "pendingQuestion": None,
                "pendingSlot": None,
                "pendingCaseType": None,    # "complaint" | "lost_and_found" | "gratitude" | "shared"
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


async def set_pending(m, channel_id: str, chat_id: str, question: Optional[str], slot: Optional[str], case_type: Optional[str]) -> None:
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {
            "pendingQuestion": question,
            "pendingSlot": slot,
            "pendingCaseType": case_type,
            "updatedAt": _now_utc(),
        }},
        upsert=True,
    )


# ----------------------------
# Cases
# ----------------------------
def _new_case_id() -> str:
    return f"KTZH-{_now_utc().strftime('%Y%m%d')}-{secrets.token_hex(4).upper()}"


async def load_case(m, case_id: str) -> Optional[Dict[str, Any]]:
    return await m.cases.find_one({"caseId": case_id})


async def load_active_case(m, sess: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cid = sess.get("activeCaseId")
    if cid:
        c = await load_case(m, cid)
        if c and c.get("status") in ("collecting", "open"):
            return c

    ac = sess.get("activeCases") or {}
    if isinstance(ac, dict):
        for k in ("complaint", "lost_and_found", "info", "gratitude"):
            if ac.get(k):
                c = await load_case(m, ac[k])
                if c and c.get("status") in ("collecting", "open"):
                    return c
    return None


async def load_active_case_by_type(m, sess: Dict[str, Any], case_type: str) -> Optional[Dict[str, Any]]:
    ac = sess.get("activeCases") or {}
    if isinstance(ac, dict) and ac.get(case_type):
        c = await load_case(m, ac[case_type])
        if c and c.get("status") in ("collecting", "open"):
            return c
    return None


async def set_active_case(m, channel_id: str, chat_id: str, case_type: str, case_id: str, make_primary: bool = True) -> None:
    patch = {f"activeCases.{case_type}": case_id, "updatedAt": _now_utc()}
    if make_primary:
        patch["activeCaseId"] = case_id
    await m.sessions.update_one({"channelId": channel_id, "chatId": chat_id}, {"$set": patch}, upsert=True)


async def close_case(m, case_id: str, status: str = "closed") -> None:
    await m.cases.update_one({"caseId": case_id}, {"$set": {"status": status, "updatedAt": _now_utc()}})


async def create_case(
    m,
    channel_id: str,
    chat_id: str,
    chat_type: str,
    contact_name: Optional[str],
    case_type: str,
    language: str = "ru",
    seed_extracted: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = _now_utc()
    case_id = _new_case_id()
    doc = {
        "caseId": case_id,
        "status": "collecting",
        "caseType": case_type,
        "channelId": channel_id,
        "chatId": chat_id,
        "chatType": chat_type,
        "contactName": contact_name,
        "language": language,
        "extracted": seed_extracted or {},
        "evidence": [],
        "attachments": [],
        "lastText": None,
        "createdAt": now,
        "updatedAt": now,
    }
    await m.cases.insert_one(doc)
    return doc


# ----------------------------
# slots logic
# ----------------------------
def _gratitude_needs_train_car(ex: Dict[str, Any]) -> bool:
    txt = (ex.get("gratitudeText") or "").lower()
    if ex.get("staffRole") or ex.get("staffName"):
        return True
    return any(w in txt for w in ["проводник", "кондуктор", "кассир", "начальник поезда", "стюард"])


def required_slots(case: Dict[str, Any]) -> List[str]:
    ct = case.get("caseType")
    ex = case.get("extracted") or {}
    missing: List[str] = []

    if ct == "complaint":
        if not ex.get("train"):
            missing.append("train")
        if not ex.get("carNumber"):
            missing.append("carNumber")
        if not ex.get("complaintText"):
            missing.append("complaintText")

    elif ct == "lost_and_found":
        if not ex.get("train"):
            missing.append("train")
        if not ex.get("carNumber"):
            missing.append("carNumber")
        if not ex.get("seat"):
            missing.append("seat")
        if not ex.get("item"):
            missing.append("item")
        if not ex.get("when"):
            missing.append("when")

    elif ct == "gratitude":
        if not ex.get("gratitudeText"):
            missing.append("gratitudeText")
        else:
            # ✅ адресная благодарность -> уточняем поезд+вагон
            if _gratitude_needs_train_car(ex):
                if not ex.get("train"):
                    missing.append("train")
                if not ex.get("carNumber"):
                    missing.append("carNumber")

    elif ct == "info":
        if not ex.get("question"):
            missing.append("question")

    return missing


def _human_prefix(case: Dict[str, Any]) -> str:
    ex = case.get("extracted") or {}
    parts = []
    if ex.get("train"):
        parts.append(f"поезд {ex['train']}")
    if ex.get("carNumber"):
        parts.append(f"вагон {ex['carNumber']}")
    if ex.get("seat") and ex.get("seat") != "UNKNOWN":
        parts.append(f"место {ex['seat']}")
    if parts:
        return "Понял(а), " + ", ".join(parts) + ". "
    return ""


def build_question(case: Dict[str, Any], slot: str) -> Tuple[str, str]:
    ct = case.get("caseType")
    pref = _human_prefix(case)

    if ct == "complaint":
        if slot == "train":
            return (pref + "Уточните номер поезда (например: Т58). Если знаете — сразу напишите и номер вагона.", "train")
        if slot == "carNumber":
            return (pref + "Уточните номер вагона. Можно вместе с поездом (Т58, вагон 8).", "carNumber")
        if slot == "complaintText":
            return (pref + "Коротко опишите суть жалобы (1–2 предложения).", "complaintText")

    if ct == "lost_and_found":
        if slot == "train":
            return (pref + "Уточните номер поезда (например: Т58). Если знаете — сразу напишите и вагон.", "train")
        if slot == "carNumber":
            return (pref + "Уточните номер вагона, где оставили вещь.", "carNumber")
        if slot == "seat":
            return (pref + "Уточните место (например: место 12). Если не помните — напишите «не помню».", "seat")
        if slot == "item":
            return (pref + "Опишите вещь: что это, цвет/размер, что внутри (если было).", "item")
        if slot == "when":
            return (pref + "Когда примерно оставили/обнаружили пропажу? (дата/время, хотя бы примерно)", "when")

    if ct == "gratitude":
        if slot == "gratitudeText":
            return ("Спасибо! Кого и за что хотите поблагодарить? (1–2 предложения)", "gratitudeText")
        if slot == "train":
            return ("Чтобы мы точно передали благодарность нужной бригаде, уточните номер поезда (например: Т58). Если знаете — можно сразу и вагон.", "train")
        if slot == "carNumber":
            return ("Уточните, пожалуйста, номер вагона (например: вагон 4).", "carNumber")

    if ct == "info":
        if slot == "question":
            return ("Уточните, пожалуйста, ваш вопрос.", "question")

    return ("Уточните, пожалуйста, детали.", slot)


# ----------------------------
# update case with message
# ----------------------------
async def update_case_with_message(
    m,
    case: Dict[str, Any],
    msg_doc: Dict[str, Any],
    nlu: Any,
    sess: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = _now_utc()
    text = (msg_doc.get("text") or "").strip()
    ex = dict(case.get("extracted") or {})
    ent = extract_entities(text)

    # pending context
    pending_slot = (sess or {}).get("pendingSlot")
    pending_case_type = (sess or {}).get("pendingCaseType")

    allow_pending = True
    if pending_case_type and pending_case_type not in (case.get("caseType"), "shared"):
        allow_pending = False

    filled_pending = False

    if pending_slot and allow_pending:
        t = text.strip()

        if pending_slot == "carNumber":
            n = _first_int(t)
            if n is not None and 1 <= n <= 99:
                ent["carNumber"] = n
                filled_pending = True

        elif pending_slot == "train":
            mt = TRAIN_RE.search(t)
            if mt:
                ent["train"] = f"T{mt.group(1)}".upper()
                filled_pending = True

        elif pending_slot == "seat":
            if t.lower() in ("не помню", "не знаю"):
                ent["seat"] = "UNKNOWN"
                filled_pending = True
            else:
                ent["seat"] = t.upper()
                filled_pending = True

        elif pending_slot in ("complaintText", "gratitudeText", "item", "when", "question"):
            if t:
                ent[pending_slot] = t
                filled_pending = True

    ct = case.get("caseType")

    # common fields
    if ent.get("train"):
        ex["train"] = ent["train"]
    if ent.get("carNumber") is not None:
        ex["carNumber"] = ent["carNumber"]

    # complaint
    if ct == "complaint":
        if ent.get("complaintText"):
            ex["complaintText"] = ent["complaintText"]
        else:
            if len(text) >= 8 and not text.isdigit():
                ex.setdefault("complaintText", text)

    # lost&found
    elif ct == "lost_and_found":
        if ent.get("seat"):
            ex["seat"] = ent["seat"]

        if ent.get("item"):
            ex["item"] = ent["item"]

        if ent.get("when"):
            ex["when"] = ent["when"]

        # попытка вытащить “вещь” из фразы “оставил/забыл …”
        if not ex.get("item"):
            mi = ITEM_AFTER_VERB_RE.search(text)
            if mi:
                item_raw = mi.group(2).strip()
                # чуть-чуть чистим
                item_raw = re.sub(r"\b(в|на)\s+вагоне.*$", "", item_raw, flags=re.IGNORECASE).strip()
                if item_raw and len(item_raw) <= 120:
                    ex["item"] = item_raw

    # gratitude
    elif ct == "gratitude":
        if ent.get("staffRole"):
            ex["staffRole"] = ent["staffRole"]
        if ent.get("staffName"):
            ex["staffName"] = ent["staffName"]

        if ent.get("gratitudeText"):
            ex["gratitudeText"] = ent["gratitudeText"]
        else:
            low = text.lower().strip()
            if low not in ("благодарность", "спасибо", "рахмет") and len(text) >= 10:
                ex["gratitudeText"] = text

    # info
    elif ct == "info":
        if ent.get("question"):
            ex["question"] = ent["question"]
        else:
            if len(text) >= 5:
                ex["question"] = text

    # evidence + attachments
    evidence = case.get("evidence") or []
    if text:
        evidence.append({"at": now, "text": text, "messageId": msg_doc.get("messageId")})

    attachments = case.get("attachments") or []
    if msg_doc.get("contentUri"):
        attachments.append({"at": now, "contentUri": msg_doc.get("contentUri"), "type": msg_doc.get("type")})

    await m.cases.update_one(
        {"caseId": case["caseId"]},
        {"$set": {
            "extracted": ex,
            "evidence": evidence[-50:],
            "attachments": attachments[-20:],
            "lastText": text or case.get("lastText"),
            "updatedAt": now,
        }},
    )

    # если мы реально получили ожидаемый слот — сбросим pending
    if filled_pending and case.get("channelId") and case.get("chatId"):
        await set_pending(m, case["channelId"], case["chatId"], None, None, None)

    updated = await m.cases.find_one({"caseId": case["caseId"]})
    return updated or case


# ----------------------------
# formatting
# ----------------------------
def format_dispatch_text(case: Dict[str, Any]) -> str:
    ct = case.get("caseType")
    ex = case.get("extracted") or {}
    lines = [
        f"Заявка: {case.get('caseId')}",
        f"Тип: {ct}",
        f"Контакт: {case.get('contactName') or '-'}",
    ]

    if ct == "complaint":
        lines += [
            f"Поезд: {ex.get('train') or '-'}",
            f"Вагон: {ex.get('carNumber') or '-'}",
            f"Описание: {ex.get('complaintText') or '-'}",
        ]

    if ct == "lost_and_found":
        lines += [
            f"Поезд: {ex.get('train') or '-'}",
            f"Вагон: {ex.get('carNumber') or '-'}",
            f"Место: {ex.get('seat') or '-'}",
            f"Вещь: {ex.get('item') or '-'}",
            f"Когда: {ex.get('when') or '-'}",
        ]

    if ct == "gratitude":
        staff = (ex.get("staffRole") or "-") + ((" " + ex.get("staffName")) if ex.get("staffName") else "")
        lines += [
            f"Поезд: {ex.get('train') or '-'}",
            f"Вагон: {ex.get('carNumber') or '-'}",
            f"Сотрудник: {staff.strip()}",
            f"Текст: {ex.get('gratitudeText') or '-'}",
        ]

    if ct == "info":
        lines.append(f"Вопрос: {ex.get('question') or '-'}")

    return "\n".join(lines)


def format_user_ack(case: Dict[str, Any]) -> str:
    ct = case.get("caseType")
    if ct == "complaint":
        return f"Принял(а) вашу жалобу. Номер заявки: {case['caseId']}."
    if ct == "lost_and_found":
        return f"Принял(а) заявку по забытым вещам. Номер заявки: {case['caseId']}."
    if ct == "gratitude":
        return "Спасибо за обратную связь! Передадим благодарность команде 🙏"
    if ct == "info":
        return f"Принял(а) ваш запрос. Номер: {case['caseId']}."
    return "Принял(а)."
