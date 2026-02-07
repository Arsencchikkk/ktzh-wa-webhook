from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


TRAIN_RE = re.compile(r"\b[тt]\s*-?\s*(\d{1,4})\b", re.IGNORECASE)
CAR_RE_1 = re.compile(r"\bвагон\s*(\d{1,2})\b", re.IGNORECASE)
CAR_RE_2 = re.compile(r"\b(\d{1,2})\s*вагон\b", re.IGNORECASE)
SEAT_RE_1 = re.compile(r"\bместо\s*(\d{1,3}[а-яa-z]?)\b", re.IGNORECASE)
SEAT_RE_2 = re.compile(r"\b(\d{1,3}[а-яa-z]?)\s*место\b", re.IGNORECASE)

DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\b")
TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b")
AGO_RE = re.compile(r"\b(\d{1,2})\s*(час(а|ов)?|мин(ут(а|ы)?)?)\s*назад\b", re.IGNORECASE)
REL_TIME_WORDS = ["сегодня", "вчера", "позавчера", "утром", "вечером", "ночью", "днем"]

ITEM_AFTER_VERB_RE = re.compile(r"\b(забыл|оставил|потерял|утерял)\b\s+(.+)$", re.IGNORECASE)

FILLER_REPLIES = {
    "говорю же", "я же сказал", "я сказал", "ты че", "алло", "понял", "понятно", "ок", "угу", "ага"
}


def _first_int(text: str) -> Optional[int]:
    m = re.search(r"(\d{1,4})", text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_when_hint(text: str) -> Optional[str]:
    t = (text or "").lower()
    d = DATE_RE.search(t)
    tm = TIME_RE.search(t)
    ago = AGO_RE.search(t)
    rel = next((w for w in REL_TIME_WORDS if w in t), None)

    parts: List[str] = []
    if d:
        parts.append(d.group(0))
    if tm:
        parts.append(tm.group(0))
    if ago:
        parts.append(ago.group(0))
    if rel:
        parts.append(rel)

    if parts:
        return ", ".join(dict.fromkeys(parts))
    return None


def _strip_noise(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[\s,.;:()\-_/]+", " ", t).strip()
    return t


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
                "activeCases": {},
                "activeCaseId": None,
                "pendingQuestion": None,
                "pendingSlot": None,
                "pendingTargets": None,
                "pendingCaseType": None,
            },
            "$set": {"chatType": chat_type, "updatedAt": now},
        },
        upsert=True,
    )
    sess = await m.sessions.find_one({"channelId": channel_id, "chatId": chat_id})
    return sess or {}


async def set_pending(
    m,
    channel_id: str,
    chat_id: str,
    question: Optional[str],
    slot: Optional[str],
    targets: Optional[List[str]],
    pending_case_type: Optional[str],
) -> None:
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {
            "pendingQuestion": question,
            "pendingSlot": slot,
            "pendingTargets": targets,
            "pendingCaseType": pending_case_type,
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
        for k in ("complaint", "lost_and_found", "gratitude", "info"):
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
    return ("Понял(а), " + ", ".join(parts) + ". ") if parts else ""


def build_question(case: Dict[str, Any], slot: str) -> Tuple[str, str]:
    ct = case.get("caseType")
    pref = _human_prefix(case)

    if ct == "complaint":
        if slot == "train":
            return (pref + "Уточните номер поезда (например: Т58). Если знаете — сразу напишите и номер вагона.", "train")
        if slot == "carNumber":
            return (pref + "Уточните номер вагона. Можно вместе с поездом (Т58, вагон 7).", "carNumber")
        if slot == "complaintText":
            return (pref + "Коротко опишите, что случилось (1–2 предложения).", "complaintText")

    if ct == "lost_and_found":
        # после intake, если всё равно не хватает — бандл
        if slot in ("seat", "item", "when"):
            q = (
                pref
                + "Чтобы помочь найти вещь, напишите одним сообщением:\n"
                  "1) место (если помните)\n"
                  "2) что за вещь и приметы (цвет/бренд/что внутри)\n"
                  "3) когда примерно оставили\n\n"
                  "Пример: «место 12, черная сумка Adidas, вчера 19:30».\n"
                  "Если место не помните — «место не помню»."
            )
            return (q, "lf_bundle")

        if slot == "train":
            return (pref + "Уточните номер поезда (например: Т58). Можно сразу: «Т58, вагон 7».", "train")
        if slot == "carNumber":
            return (pref + "Уточните номер вагона, где оставили вещь.", "carNumber")

    if ct == "gratitude":
        if slot == "gratitudeText":
            return ("Спасибо! Кого и за что хотите поблагодарить? (1–2 предложения)", "gratitudeText")

    if ct == "info":
        if slot == "question":
            return ("Уточните, пожалуйста, ваш вопрос.", "question")

    return ("Уточните, пожалуйста, детали.", slot)


def _allow_pending(sess: Optional[Dict[str, Any]], case_type: str) -> bool:
    if not sess:
        return False
    targets = sess.get("pendingTargets")
    if isinstance(targets, list) and targets:
        return case_type in targets
    pct = sess.get("pendingCaseType")
    if pct == "shared":
        return True
    if pct and pct != case_type:
        return False
    return True


def _parse_lf_bundle(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    low = t.lower()
    out: Dict[str, Any] = {}

    if not t:
        return out

    if ("место" in low and ("не помню" in low or "не знаю" in low)) or low.strip() in ("не помню", "не знаю"):
        out["seat"] = "UNKNOWN"

    parts = [p.strip() for p in re.split(r"[;,]+", t) if p.strip()]
    if not parts:
        parts = [t]

    # seat
    ms = SEAT_RE_1.search(t) or SEAT_RE_2.search(t)
    if ms:
        out["seat"] = ms.group(1).upper()
    else:
        first = parts[0]
        if first and first[:3].strip().isdigit() and ("вагон" not in first.lower()) and (TRAIN_RE.search(first) is None):
            n = _first_int(first)
            if n is not None and 1 <= n <= 199:
                out["seat"] = str(n)

    # when
    wh = _extract_when_hint(t)
    if wh:
        out["when"] = wh

    # item
    mi = ITEM_AFTER_VERB_RE.search(t)
    if mi:
        cand = _strip_noise(mi.group(2))
        if cand and cand.lower() not in FILLER_REPLIES:
            out["item"] = cand
            return out

    item_chunks: List[str] = []
    for p in parts:
        pl = p.lower()
        if pl in FILLER_REPLIES:
            continue
        if DATE_RE.search(p) or TIME_RE.search(p) or AGO_RE.search(p) or any(w in pl for w in REL_TIME_WORDS):
            continue
        if "место" in pl:
            continue
        if p.isdigit() and "seat" not in out:
            continue
        item_chunks.append(p)

    cand2 = _strip_noise(" ".join(item_chunks))
    if cand2 and cand2.lower() not in FILLER_REPLIES and cand2.lower() not in ("не помню", "не знаю"):
        out["item"] = cand2

    return out


def _apply_pending_overrides(text: str, pending_slot: Optional[str]) -> Dict[str, Any]:
    t = (text or "").strip()
    if not pending_slot or not t:
        return {}

    out: Dict[str, Any] = {}

    if pending_slot == "carNumber":
        n = _first_int(t)
        if n is not None and 1 <= n <= 99:
            out["carNumber"] = n
        return out

    if pending_slot == "train":
        mt = TRAIN_RE.search(t)
        if mt:
            out["train"] = f"T{mt.group(1)}".upper()
        return out

    if pending_slot == "train_car":
        mt = TRAIN_RE.search(t)
        if mt:
            out["train"] = f"T{mt.group(1)}".upper()
        mc = CAR_RE_1.search(t) or CAR_RE_2.search(t)
        if mc:
            out["carNumber"] = int(mc.group(1))
        else:
            # "Т58, 7"
            nums = re.findall(r"\d{1,4}", t)
            if mt and len(nums) >= 2:
                train_num = mt.group(1)
                for x in nums:
                    if x != train_num:
                        try:
                            n = int(x)
                            if 1 <= n <= 99:
                                out["carNumber"] = n
                                break
                        except Exception:
                            pass
        return out

    if pending_slot == "lf_bundle":
        return _parse_lf_bundle(t)

    # ✅ универсальный intake: поезд+вагон+место+вещь+когда одним сообщением
    if pending_slot == "lf_intake":
        out.update(_apply_pending_overrides(t, "train_car"))
        out.update(_parse_lf_bundle(t))
        # если item не вытащился, но есть "потерял/забыл/оставил ..." — возьмем хвост
        if "item" not in out:
            mi = ITEM_AFTER_VERB_RE.search(t)
            if mi:
                cand = _strip_noise(mi.group(2))
                if cand:
                    out["item"] = cand
        return out

    if pending_slot in ("complaintText", "gratitudeText", "item", "when", "question"):
        low = t.lower().strip()
        if low in FILLER_REPLIES:
            return {}
        out[pending_slot] = t
        return out

    return {}


async def update_case_with_message(
    m,
    case: Dict[str, Any],
    msg_doc: Dict[str, Any],
    nlu: Any,
    sess: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = _now_utc()
    text = (msg_doc.get("text") or "").strip()
    low = text.lower().strip()

    ex = dict(case.get("extracted") or {})
    ent = extract_entities(text)

    pending_slot = (sess or {}).get("pendingSlot")
    used_pending = False

    if _allow_pending(sess, case.get("caseType", "")):
        extra = _apply_pending_overrides(text, pending_slot)
        if extra:
            ent.update(extra)
            used_pending = True

    # shared
    if ent.get("train"):
        ex["train"] = ent["train"]
    if ent.get("carNumber") is not None:
        ex["carNumber"] = ent["carNumber"]

    ct = case.get("caseType")

    if ct == "complaint":
        if ent.get("complaintText"):
            ex["complaintText"] = ent["complaintText"]
        else:
            if text and not text.isdigit() and low not in FILLER_REPLIES and len(text) >= 8:
                ex.setdefault("complaintText", text)

    elif ct == "lost_and_found":
        if ent.get("seat"):
            ex["seat"] = ent["seat"]
        if ent.get("item"):
            ex["item"] = ent["item"]
        if ent.get("when"):
            ex["when"] = ent["when"]

        if not ex.get("item") and text:
            mi = ITEM_AFTER_VERB_RE.search(text)
            if mi:
                cand = _strip_noise(mi.group(2))
                if cand and cand.lower() not in FILLER_REPLIES:
                    ex["item"] = cand

        if not ex.get("when"):
            wh = _extract_when_hint(text)
            if wh:
                ex["when"] = wh

    elif ct == "gratitude":
        if ent.get("gratitudeText"):
            ex["gratitudeText"] = ent["gratitudeText"]
        else:
            if low in ("благодарность", "спасибо", "рахмет"):
                pass
            elif len(text) >= 10 and low not in FILLER_REPLIES:
                ex["gratitudeText"] = text

    elif ct == "info":
        if ent.get("question"):
            ex["question"] = ent["question"]
        else:
            if len(text) >= 5 and low not in FILLER_REPLIES:
                ex["question"] = text

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

    # ✅ если мы реально обработали pending — чистим его, чтобы не было “повторите вагон…”
    if used_pending and case.get("channelId") and case.get("chatId"):
        await set_pending(m, case["channelId"], case["chatId"], None, None, None, None)

    updated = await m.cases.find_one({"caseId": case["caseId"]})
    return updated or case


def format_dispatch_text(case: Dict[str, Any]) -> str:
    ct = case.get("caseType")
    ex = case.get("extracted") or {}
    lines = [f"Заявка: {case.get('caseId')}", f"Тип: {ct}", f"Контакт: {case.get('contactName') or '-'}"]

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
        lines.append(f"Текст: {ex.get('gratitudeText') or '-'}")

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
