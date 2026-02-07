from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# time helpers (ALWAYS tz-aware UTC)
# ----------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ----------------------------
# lightweight entity extraction
# ----------------------------
TRAIN_RE = re.compile(r"\b[тt]\s*-?\s*(\d{1,4})\b", re.IGNORECASE)
CAR_RE = re.compile(r"\bвагон\s*(\d{1,2})\b", re.IGNORECASE)
SEAT_RE = re.compile(r"\bместо\s*(\d{1,3}[а-яa-z]?)\b", re.IGNORECASE)

LOST_KEYWORDS = ("забыл", "оставил", "потерял", "утерял", "сумк", "рюкзак", "телефон", "кошелек", "паспорт", "вещ")


def extract_entities(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    out: Dict[str, Any] = {}

    m = TRAIN_RE.search(t)
    if m:
        out["train"] = f"T{m.group(1)}".upper()

    m = CAR_RE.search(t)
    if m:
        out["carNumber"] = int(m.group(1))

    m = SEAT_RE.search(t)
    if m:
        out["seat"] = m.group(1).upper()

    # если просто цифра — это может быть вагон/место, но решаем по pendingSlot
    return out


# ----------------------------
# Session
# ----------------------------
async def ensure_session(m, channel_id: str, chat_id: str, chat_type: str) -> Dict[str, Any]:
    """
    ВАЖНО: НЕ писать одни и те же поля в $setOnInsert и $set.
    """
    now = _now_utc()
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {
            "$setOnInsert": {
                "channelId": channel_id,
                "chatId": chat_id,
                "createdAt": now,
                "draftSlots": {},
                "activeCases": {},     # {"complaint": caseId, "lost_and_found": caseId, ...}
                "pendingQuestion": None,
                "pendingSlot": None,
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


async def _set_pending(m, channel_id: str, chat_id: str, question: Optional[str], slot: Optional[str]) -> None:
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {"pendingQuestion": question, "pendingSlot": slot, "updatedAt": _now_utc()}},
        upsert=True,
    )


async def _update_draft_slots(m, channel_id: str, chat_id: str, patch: Dict[str, Any]) -> None:
    if not patch:
        return
    # draftSlots.x = val
    upd = {f"draftSlots.{k}": v for k, v in patch.items()}
    upd["updatedAt"] = _now_utc()
    await m.sessions.update_one({"channelId": channel_id, "chatId": chat_id}, {"$set": upd}, upsert=True)


# ----------------------------
# Cases
# ----------------------------
def _new_case_id() -> str:
    return f"KTZH-{_now_utc().strftime('%Y%m%d')}-{secrets.token_hex(4).upper()}"


async def load_case(m, case_id: str) -> Optional[Dict[str, Any]]:
    return await m.cases.find_one({"caseId": case_id})


async def load_active_case(m, sess: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Возвращает "главный" активный кейс (если есть).
    """
    # backward compatibility
    cid = sess.get("activeCaseId")
    if cid:
        return await load_case(m, cid)

    ac = sess.get("activeCases") or {}
    if isinstance(ac, dict):
        # приоритет: complaint -> lost -> info -> gratitude
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
    patch = {
        f"activeCases.{case_type}": case_id,
        "updatedAt": _now_utc(),
    }
    if make_primary:
        patch["activeCaseId"] = case_id  # optional primary pointer
    await m.sessions.update_one({"channelId": channel_id, "chatId": chat_id}, {"$set": patch}, upsert=True)


async def close_case(m, case_id: str, status: str = "closed") -> None:
    await m.cases.update_one(
        {"caseId": case_id},
        {"$set": {"status": status, "updatedAt": _now_utc()}},
    )


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

    extracted = seed_extracted or {}
    doc = {
        "caseId": case_id,
        "status": "collecting",
        "caseType": case_type,
        "channelId": channel_id,
        "chatId": chat_id,
        "chatType": chat_type,
        "contactName": contact_name,
        "language": language,
        "extracted": extracted,
        "evidence": [],
        "attachments": [],
        "lastText": None,
        "createdAt": now,
        "updatedAt": now,
    }
    await m.cases.insert_one(doc)
    return doc


def _normalize_free_text(text: str) -> str:
    return (text or "").strip()


def required_slots(case: Dict[str, Any]) -> List[str]:
    ct = case.get("caseType")
    ex = case.get("extracted") or {}
    missing: List[str] = []

    if ct == "complaint":
        # по жалобе просим поезд и вагон ASAP
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
        # одно слово "Благодарность" — недостаточно
        if not ex.get("gratitudeText"):
            missing.append("gratitudeText")

    elif ct == "info":
        if not ex.get("question"):
            missing.append("question")

    return missing


def build_question(case: Dict[str, Any], slot: str) -> Tuple[str, str]:
    ct = case.get("caseType")

    if ct == "complaint":
        if slot == "train":
            return ("Уточните, пожалуйста, номер поезда (например: Т58).", "train")
        if slot == "carNumber":
            return ("Уточните, пожалуйста, номер вагона.", "carNumber")
        if slot == "complaintText":
            return ("Опишите, пожалуйста, подробнее, что именно случилось (1–2 предложения).", "complaintText")

    if ct == "lost_and_found":
        if slot == "train":
            return ("Уточните номер поезда (например: Т58).", "train")
        if slot == "carNumber":
            return ("Уточните номер вагона, где оставили вещь.", "carNumber")
        if slot == "seat":
            return ("Уточните место (например: место 12) — если не помните, напишите «не помню».", "seat")
        if slot == "item":
            return ("Опишите вещь: что это, цвет/размер, что внутри (если было).", "item")
        if slot == "when":
            return ("Когда примерно оставили/обнаружили пропажу? (дата/время, хотя бы примерно)", "when")

    if ct == "gratitude":
        if slot == "gratitudeText":
            return ("Спасибо! Кого и за что хотите поблагодарить? (1–2 предложения)", "gratitudeText")

    if ct == "info":
        if slot == "question":
            return ("Уточните, пожалуйста, ваш вопрос.", "question")

    return ("Уточните, пожалуйста, детали.", slot)


async def update_case_with_message(
    m,
    case: Dict[str, Any],
    msg_doc: Dict[str, Any],
    nlu: Any,
    sess: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Обновляет extracted + evidence + attachments.
    Также учитывает pendingSlot из session и draftSlots.
    """
    now = _now_utc()

    text = _normalize_free_text(msg_doc.get("text") or "")
    ex = dict(case.get("extracted") or {})

    # 1) базовая экстракция из текста
    ent = extract_entities(text)

    # 2) если есть pendingSlot — пытаемся интерпретировать ответ
    pending_slot = (sess or {}).get("pendingSlot")
    if pending_slot:
        t = text.strip()
        if pending_slot == "carNumber" and t.isdigit():
            ent["carNumber"] = int(t)
        elif pending_slot == "train":
            mtrain = TRAIN_RE.search(t)
            if mtrain:
                ent["train"] = f"T{mtrain.group(1)}".upper()
        elif pending_slot == "seat":
            # место может быть "12", "12А", или "не помню"
            if t.lower() in ("не помню", "не знаю"):
                ent["seat"] = "UNKNOWN"
            else:
                ent["seat"] = t.upper()
        elif pending_slot in ("complaintText", "gratitudeText", "item", "when", "question"):
            ent[pending_slot] = t

    # 3) подтягиваем из nlu (если он у тебя умеет)
    # (бережно: только если поле есть)
    if getattr(nlu, "language", None):
        ex["language"] = nlu.language

    # 4) заполняем extracted по типу кейса
    ct = case.get("caseType")
    if ct == "complaint":
        if ent.get("train"):
            ex["train"] = ent["train"]
        if ent.get("carNumber") is not None:
            ex["carNumber"] = ent["carNumber"]

        # complaintText: если текст не короткий и не чисто "Т58"/"5"
        if "complaintText" in ent:
            ex["complaintText"] = ent["complaintText"]
        else:
            # если в сообщении уже есть суть жалобы — сохраняем
            if len(text) >= 8 and not TRAIN_RE.fullmatch(text.strip()) and not text.strip().isdigit():
                ex["complaintText"] = text

    elif ct == "lost_and_found":
        if ent.get("train"):
            ex["train"] = ent["train"]
        if ent.get("carNumber") is not None:
            ex["carNumber"] = ent["carNumber"]
        if ent.get("seat"):
            ex["seat"] = ent["seat"]
        if ent.get("item"):
            ex["item"] = ent["item"]
        if ent.get("when"):
            ex["when"] = ent["when"]

        # если сообщение содержит “оставил/забыл …” — можно записать item как весь текст, если item пустой
        if not ex.get("item") and any(k in text.lower() for k in LOST_KEYWORDS) and len(text) > 8:
            ex["item"] = text

    elif ct == "gratitude":
        if "gratitudeText" in ent:
            ex["gratitudeText"] = ent["gratitudeText"]
        else:
            # если это не просто "благодарность"
            if len(text) >= 10 and text.lower().strip() not in ("благодарность", "спасибо", "рахмет"):
                ex["gratitudeText"] = text

    elif ct == "info":
        if "question" in ent:
            ex["question"] = ent["question"]
        else:
            if len(text) >= 5:
                ex["question"] = text

    # 5) evidence & attachments
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
            "evidence": evidence[-50:],   # ограничим
            "attachments": attachments[-20:],
            "lastText": text or case.get("lastText"),
            "updatedAt": now,
        }},
    )
    case = await m.cases.find_one({"caseId": case["caseId"]})
    return case or case


def format_dispatch_text(case: Dict[str, Any]) -> str:
    ct = case.get("caseType")
    ex = case.get("extracted") or {}
    lines = [f"Заявка: {case.get('caseId')}", f"Тип: {ct}", f"Контакт: {case.get('contactName') or '-'}"]

    if ct == "complaint":
        lines.append(f"Поезд: {ex.get('train') or '-'}")
        lines.append(f"Вагон: {ex.get('carNumber') or '-'}")
        lines.append(f"Описание: {ex.get('complaintText') or '-'}")

    if ct == "lost_and_found":
        lines.append(f"Поезд: {ex.get('train') or '-'}")
        lines.append(f"Вагон: {ex.get('carNumber') or '-'}")
        lines.append(f"Место: {ex.get('seat') or '-'}")
        lines.append(f"Вещь: {ex.get('item') or '-'}")
        lines.append(f"Когда: {ex.get('when') or '-'}")

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


# ----------------------------
# Combined question for 2 cases
# ----------------------------
def build_combined_question(missing_by_type: Dict[str, List[str]]) -> Tuple[str, str]:
    """
    Возвращает один человеческий вопрос, чтобы не спамить.
    И pendingSlot = первый слот, который ждём (упрощение).
    """
    # приоритет вопросов: поезд -> вагон -> место -> описание
    order = ["train", "carNumber", "seat", "complaintText", "item", "when", "gratitudeText", "question"]

    # найдём первый слот в порядке
    chosen_slot = None
    chosen_case_type = None
    for s in order:
        for ct, miss in missing_by_type.items():
            if s in miss:
                chosen_slot = s
                chosen_case_type = ct
                break
        if chosen_slot:
            break

    if not chosen_slot:
        return ("Уточните, пожалуйста, детали.", "details")

    # текст вопроса
    if chosen_slot == "train":
        return ("Уточните номер поезда (например: Т58).", "train")
    if chosen_slot == "carNumber":
        return ("Уточните номер вагона.", "carNumber")
    if chosen_slot == "seat":
        return ("Уточните место (например: место 12) — если не помните, напишите «не помню».", "seat")
    if chosen_slot == "complaintText":
        return ("Коротко опишите суть жалобы (1–2 предложения).", "complaintText")
    if chosen_slot == "item":
        return ("Опишите забытую вещь: что это, цвет/размер, что внутри.", "item")
    if chosen_slot == "when":
        return ("Когда примерно оставили/обнаружили пропажу? (дата/время, примерно)", "when")
    if chosen_slot == "gratitudeText":
        return ("Кого и за что хотите поблагодарить? (1–2 предложения)", "gratitudeText")
    if chosen_slot == "question":
        return ("Уточните, пожалуйста, ваш вопрос.", "question")

    return ("Уточните, пожалуйста, детали.", chosen_slot)
