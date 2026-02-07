from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Tuple

from . import settings
from .routing import resolve_region, resolve_executor
from .llm_nlu import llm_extract, sanitize_slots


# ----------------------------
# time helpers
# ----------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


PENDING_TTL = timedelta(minutes=15)


# ----------------------------
# rule-based helpers (guardrails)
# ----------------------------
GREET_RX = re.compile(r"^\s*(здравствуй(те)?|привет|добрый\s*(день|вечер)|сәлем(етсіз\s*бе)?|саламат(сыз\s*ба)?)\s*[!.]*\s*$", re.IGNORECASE)

GRAT_WORDS = ["благодар", "спасибо", "рахмет", "алғыс", "мерси"]
LOST_WORDS = ["забыл", "оставил", "потерял", "утерял", "пропал", "сумк", "рюкзак", "кошел", "паспорт", "телефон", "вещ", "термос"]
COMPLAINT_WORDS = ["жалоб", "плох", "ужас", "хам", "груб", "гряз", "туалет", "бумаг", "вода", "жарк", "холод", "опозд", "задерж", "сервис", "не работает"]
FOUND_WORDS = ["нашлась", "нашел", "нашёл", "нашли", "found", "табылды"]

TRAIN_RE = re.compile(r"\b[тt]\s*-?\s*(\d{1,4})\b", re.IGNORECASE)
WAGON_RE_1 = re.compile(r"\bвагон\s*(\d{1,2})\b", re.IGNORECASE)
WAGON_RE_2 = re.compile(r"\b(\d{1,2})\s*вагон\b", re.IGNORECASE)

TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _has_any(text: str, words: List[str]) -> bool:
    t = _norm(text)
    return any(w in t for w in words)


def meaning_score(text: str) -> int:
    t = _norm(text)
    if not t:
        return 0
    if GREET_RX.match(text):
        return 0
    if t in ("ок", "okay", "да", "нет", "угу", "ага", "."):
        return 0
    if t.isdigit():
        return 0

    score = 0
    if len(t) >= 8:
        score += 1
    if len(t) >= 25:
        score += 1
    if _has_any(t, LOST_WORDS):
        score += 2
    if _has_any(t, COMPLAINT_WORDS):
        score += 2
    if _has_any(t, GRAT_WORDS):
        score += 1
    if "?" in t:
        score += 1
    return score


def extract_entities_rule(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    out: Dict[str, Any] = {}

    mt = TRAIN_RE.search(t)
    if mt:
        out["train"] = f"T{mt.group(1)}".upper()

    mw = WAGON_RE_1.search(t) or WAGON_RE_2.search(t)
    if mw:
        try:
            out["carNumber"] = int(mw.group(1))
        except Exception:
            pass

    tm = TIME_RE.search(t)
    if tm:
        out["when"] = tm.group(0)

    return out


# ----------------------------
# Mongo: sessions/cases
# ----------------------------
def _new_case_id() -> str:
    return f"KTZH-{_now_utc().strftime('%Y%m%d')}-{secrets.token_hex(4).upper()}"


async def ensure_session(m, channel_id: str, chat_id: str, chat_type: str) -> Dict[str, Any]:
    now = _now_utc()
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$setOnInsert": {
            "channelId": channel_id,
            "chatId": chat_id,
            "createdAt": now,
            "activeCases": {},
            "sharedSlots": {},
            "pending": None,  # {"slots": [...], "caseTypes": [...], "question": str, "at": dt}
        }, "$set": {"chatType": chat_type, "updatedAt": now}},
        upsert=True,
    )
    return await m.sessions.find_one({"channelId": channel_id, "chatId": chat_id}) or {}


async def set_pending(m, channel_id: str, chat_id: str, question: str, slots: List[str], case_types: List[str]) -> None:
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {
            "pending": {"question": question, "slots": slots, "caseTypes": case_types, "at": _now_utc()},
            "updatedAt": _now_utc(),
        }},
        upsert=True,
    )


async def clear_pending(m, channel_id: str, chat_id: str) -> None:
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {"pending": None, "updatedAt": _now_utc()}},
        upsert=True,
    )


async def set_shared_slots(m, channel_id: str, chat_id: str, patch: Dict[str, Any]) -> None:
    if not patch:
        return
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {**{f"sharedSlots.{k}": v for k, v in patch.items()}, "updatedAt": _now_utc()}},
        upsert=True,
    )


async def set_active_case(m, channel_id: str, chat_id: str, case_type: str, case_id: str) -> None:
    await m.sessions.update_one(
        {"channelId": channel_id, "chatId": chat_id},
        {"$set": {f"activeCases.{case_type}": case_id, "updatedAt": _now_utc()}},
        upsert=True,
    )


async def load_case(m, case_id: str) -> Optional[Dict[str, Any]]:
    return await m.cases.find_one({"caseId": case_id})


async def load_active_cases(m, sess: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    ac = sess.get("activeCases") or {}
    if not isinstance(ac, dict):
        return out
    for ct, cid in ac.items():
        if not cid:
            continue
        c = await load_case(m, cid)
        if c and c.get("status") == "collecting":
            out[ct] = c
    return out


async def get_or_create_case(m, sess: Dict[str, Any], channel_id: str, chat_id: str, chat_type: str, contact_name: Optional[str], case_type: str) -> Dict[str, Any]:
    active = await load_active_cases(m, sess)
    if case_type in active:
        return active[case_type]

    shared = sess.get("sharedSlots") if isinstance(sess.get("sharedSlots"), dict) else {}
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
        "extracted": dict(shared or {}),
        "categories": [],
        "severity_1_5": None,
        "evidence": [],
        "attachments": [],
        "createdAt": now,
        "updatedAt": now,
    }
    await m.cases.insert_one(doc)
    await set_active_case(m, channel_id, chat_id, case_type, case_id)
    return doc


# ----------------------------
# requirements (slots)
# ----------------------------
def required_slots(case: Dict[str, Any]) -> List[str]:
    ct = case.get("caseType")
    ex = case.get("extracted") or {}
    miss: List[str] = []

    if ct == "complaint":
        if not ex.get("train"):
            miss.append("train")
        if not ex.get("carNumber"):
            miss.append("carNumber")
        if not ex.get("complaintText"):
            miss.append("complaintText")

    elif ct == "lost_and_found":
        if not ex.get("train"):
            miss.append("train")
        if not ex.get("carNumber"):
            miss.append("carNumber")
        if not ex.get("place"):
            miss.append("place")
        if not ex.get("item"):
            miss.append("item")
        if ex.get("item") and len(_norm(ex.get("item"))) <= 8 and not ex.get("itemDetails"):
            miss.append("itemDetails")
        if not ex.get("when"):
            miss.append("when")

    elif ct == "gratitude":
        if not ex.get("gratitudeText"):
            miss.append("gratitudeText")

    elif ct == "info":
        if not ex.get("question"):
            miss.append("question")

    return miss


def build_question_for(missing_by_type: Dict[str, List[str]]) -> Tuple[str, List[str], List[str]]:
    """
    Возвращает 1 вопрос на ход: (question, slots, caseTypes)
    """
    # shared train+car
    need_train = any("train" in miss for miss in missing_by_type.values())
    need_car = any("carNumber" in miss for miss in missing_by_type.values())
    if need_train and need_car:
        return (
            "Уточните, пожалуйста, номер поезда и номер вагона (пример: Т58, вагон 7).",
            ["train", "carNumber"],
            list(missing_by_type.keys()),
        )
    if need_train:
        return ("Уточните номер поезда (например: Т58).", ["train"], list(missing_by_type.keys()))
    if need_car:
        return ("Уточните номер вагона.", ["carNumber"], list(missing_by_type.keys()))

    # lost bundle
    if "lost_and_found" in missing_by_type:
        miss = missing_by_type["lost_and_found"]
        # пакетно
        if any(x in miss for x in ("place", "item", "itemDetails", "when")):
            return (
                "Чтобы помочь найти вещь, напишите ОДНИМ сообщением:\n"
                "• где оставили (место/купе/полка/тамбур)\n"
                "• что за вещь и признаки (цвет/бренд/что внутри)\n"
                "• когда примерно\n"
                "Пример: «место 12, черная сумка Adidas, вчера 19:30».",
                ["place", "item", "itemDetails", "when"],
                ["lost_and_found"],
            )

    # complaint
    if "complaint" in missing_by_type and "complaintText" in missing_by_type["complaint"]:
        return ("Коротко опишите суть жалобы (1–2 предложения).", ["complaintText"], ["complaint"])

    # gratitude
    if "gratitude" in missing_by_type and "gratitudeText" in missing_by_type["gratitude"]:
        return ("Кого и за что хотите поблагодарить? (1–2 предложения).", ["gratitudeText"], ["gratitude"])

    # info
    if "info" in missing_by_type and "question" in missing_by_type["info"]:
        return ("Уточните, пожалуйста, ваш вопрос.", ["question"], ["info"])

    return ("Уточните, пожалуйста, детали.", [], list(missing_by_type.keys()))


def case_ready(case: Dict[str, Any]) -> bool:
    ct = case.get("caseType")
    ex = case.get("extracted") or {}
    if ct == "complaint":
        return bool(ex.get("train") and ex.get("carNumber") and ex.get("complaintText"))
    if ct == "lost_and_found":
        return bool(ex.get("train") and ex.get("carNumber") and ex.get("place") and ex.get("item") and ex.get("when"))
    if ct == "gratitude":
        return bool(ex.get("gratitudeText"))
    if ct == "info":
        return bool(ex.get("question"))
    return False


# ----------------------------
# update case with message (merge rule+LLM)
# ----------------------------
def _apply_pending_to_entities(text: str, pending_slots: List[str]) -> Dict[str, Any]:
    t = (text or "").strip()
    out: Dict[str, Any] = {}

    if "train" in pending_slots:
        mt = TRAIN_RE.search(t)
        if mt:
            out["train"] = f"T{mt.group(1)}".upper()

    if "carNumber" in pending_slots:
        mw = WAGON_RE_1.search(t) or WAGON_RE_2.search(t)
        if mw:
            out["carNumber"] = int(mw.group(1))
        else:
            if t.isdigit():
                n = int(t)
                if 1 <= n <= 99:
                    out["carNumber"] = n

    # bundle lost
    if any(s in pending_slots for s in ("place", "item", "itemDetails", "when")):
        # очень простой парсер, LLM обычно лучше, но это fallback
        low = _norm(t)
        if "место" in low or "купе" in low or "полк" in low or "тамбур" in low or "туалет" in low:
            out["place"] = t

        if len(t) >= 6 and not t.isdigit():
            # если явно "когда" — time
            tm = TIME_RE.search(t)
            if tm:
                out["when"] = tm.group(0)
            else:
                # иначе это может быть item или details
                out.setdefault("item", t)

    for s in ("complaintText", "gratitudeText", "question"):
        if s in pending_slots and t:
            out[s] = t

    return out


def _merge_slots(ex: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(ex or {})
    for k, v in (patch or {}).items():
        if v is None:
            continue
        out[k] = v
    return out


async def update_case(
    m,
    case: Dict[str, Any],
    *,
    message_id: str,
    text: str,
    msg_type: str,
    content_uri: Optional[str],
    sess: Dict[str, Any],
    llm_slots: Optional[Dict[str, Any]] = None,
    llm_categories: Optional[List[str]] = None,
    llm_severity: Optional[int] = None,
) -> Dict[str, Any]:
    now = _now_utc()
    ex = dict(case.get("extracted") or {})
    t = (text or "").strip()

    # rule ents
    ents = extract_entities_rule(t)

    # pending ents
    pending = sess.get("pending") or None
    if isinstance(pending, dict):
        at = pending.get("at")
        if isinstance(at, datetime) and (_now_utc() - at) <= PENDING_TTL:
            ents = _merge_slots(ents, _apply_pending_to_entities(t, pending.get("slots") or []))

    # llm ents (приоритет: train/car из shared могут быть из LLM, но валидируем)
    if llm_slots:
        ents = _merge_slots(ents, llm_slots)

    # apply
    ex = _merge_slots(ex, ents)

    # complaint-specific: не принимать "Т58, 7 вагон" как complaintText
    if case["caseType"] == "complaint":
        if "complaintText" not in ex and llm_slots and llm_slots.get("complaintText"):
            ex["complaintText"] = llm_slots["complaintText"]
        if (not ex.get("complaintText")) and _has_any(t, COMPLAINT_WORDS) and len(_norm(t)) >= 10:
            # если текст не выглядит чисто сервисным
            if not (TRAIN_RE.search(t) and (WAGON_RE_1.search(t) or WAGON_RE_2.search(t)) and len(_norm(t).split()) <= 4):
                ex["complaintText"] = t

    # lost-specific
    if case["caseType"] == "lost_and_found":
        # если item слишком общий — просим details позже
        if ex.get("item") and len(_norm(ex["item"])) <= 8 and not ex.get("itemDetails") and llm_slots and llm_slots.get("itemDetails"):
            ex["itemDetails"] = llm_slots["itemDetails"]

    # evidence/attachments
    evidence = list(case.get("evidence") or [])
    if t:
        evidence.append({"at": now, "text": t, "messageId": message_id})

    attachments = list(case.get("attachments") or [])
    if content_uri:
        attachments.append({"at": now, "type": msg_type, "contentUri": content_uri, "messageId": message_id})

    patch: Dict[str, Any] = {
        "extracted": ex,
        "evidence": evidence[-60:],
        "attachments": attachments[-20:],
        "updatedAt": now,
    }

    # complaint meta
    if case["caseType"] == "complaint":
        if llm_categories is not None:
            patch["categories"] = llm_categories
        if llm_severity is not None:
            patch["severity_1_5"] = llm_severity

    await m.cases.update_one({"caseId": case["caseId"]}, {"$set": patch})
    return await m.cases.find_one({"caseId": case["caseId"]}) or case


# ----------------------------
# dispatch formatting
# ----------------------------
def format_dispatch_text(case: Dict[str, Any]) -> str:
    ct = case.get("caseType")
    ex = case.get("extracted") or {}
    cats = ", ".join(case.get("categories") or []) or "-"
    sev = case.get("severity_1_5") or "-"

    lines = [
        f"Заявка: {case.get('caseId')}",
        f"Тип: {ct}",
        f"Клиент: {case.get('chatId')}",
        f"Контакт: {case.get('contactName') or '-'}",
        f"Поезд: {ex.get('train') or '-'}",
        f"Вагон: {ex.get('carNumber') or '-'}",
    ]

    if ct == "complaint":
        lines += [
            f"Категории: {cats}",
            f"Серьёзность (1-5): {sev}",
            f"Описание: {ex.get('complaintText') or '-'}",
        ]

    if ct == "lost_and_found":
        lines += [
            f"Где: {ex.get('place') or '-'}",
            f"Вещь: {ex.get('item') or '-'}",
            f"Детали: {ex.get('itemDetails') or '-'}",
            f"Когда: {ex.get('when') or '-'}",
        ]

    if ct == "gratitude":
        lines += [
            f"Благодарность: {ex.get('gratitudeText') or '-'}",
        ]

    if ct == "info":
        lines += [
            f"Вопрос: {ex.get('question') or '-'}",
        ]

    return "\n".join(lines)


def format_user_ack(case_ids_by_type: Dict[str, str]) -> str:
    parts = []
    if "complaint" in case_ids_by_type:
        parts.append(f"Жалоба принята. Номер: {case_ids_by_type['complaint']}.")
    if "lost_and_found" in case_ids_by_type:
        parts.append(f"Заявка по забытым вещам принята. Номер: {case_ids_by_type['lost_and_found']}.")
    if "gratitude" in case_ids_by_type:
        parts.append("Спасибо! Передадим вашу благодарность 🙏")
    if "info" in case_ids_by_type:
        parts.append(f"Запрос принят. Номер: {case_ids_by_type['info']}.")
    return " ".join(parts) if parts else "Принял(а)."


# ----------------------------
# MAIN message handler
# ----------------------------
async def handle_inbound_message(
    *,
    m,
    send_text: Callable[[str, str, str, str, str], Any],
    channel_id: str,
    chat_id: str,
    chat_type: str,
    contact_name: Optional[str],
    message_id: str,
    text: str,
    msg_type: str,
    content_uri: Optional[str],
    message_dt: datetime,
) -> None:
    sess = await ensure_session(m, channel_id, chat_id, chat_type)

    # 1) pending ttl cleanup
    pending = sess.get("pending")
    if isinstance(pending, dict):
        at = pending.get("at")
        if isinstance(at, datetime) and (_now_utc() - at) > PENDING_TTL:
            await clear_pending(m, channel_id, chat_id)
            sess = await ensure_session(m, channel_id, chat_id, chat_type)

    t = (text or "").strip()
    low = _norm(t)

    # 2) greeting-only
    if GREET_RX.match(t):
        await send_text(
            channel_id, chat_type, chat_id,
            "Здравствуйте! Чем могу помочь? Можно: оставить жалобу, сообщить о забытых вещах или написать благодарность.",
            f"bot-greet-{message_id}",
        )
        return

    # 3) found -> close lost
    if _has_any(low, FOUND_WORDS):
        active = await load_active_cases(m, sess)
        lf = active.get("lost_and_found")
        if lf:
            await m.cases.update_one({"caseId": lf["caseId"]}, {"$set": {"status": "closed", "updatedAt": _now_utc()}})
        await clear_pending(m, channel_id, chat_id)
        await send_text(channel_id, chat_type, chat_id, "Понял(а), отлично что нашлось. Если нужна помощь — напишите.", f"bot-found-{message_id}")
        return

    # 4) load active collecting cases
    active = await load_active_cases(m, sess)

    # 5) if pending exists -> treat as continuation, do not run LLM
    if isinstance(sess.get("pending"), dict):
        target_case_types = sess["pending"].get("caseTypes") or list(active.keys())
    else:
        target_case_types = []

    # 6) decide case types (rule-based baseline)
    ms = meaning_score(t)

    has_grat = _has_any(low, GRAT_WORDS) or low in ("благодарность", "спасибо", "рахмет")
    has_lost = _has_any(low, LOST_WORDS)
    has_compl = _has_any(low, COMPLAINT_WORDS)

    # hard override: "благодарность" не должна стать жалобой
    if has_grat and not has_lost and not has_compl:
        baseline_types = ["gratitude"]
    else:
        if has_compl and has_lost:
            baseline_types = ["complaint", "lost_and_found"]
        elif has_lost:
            baseline_types = ["lost_and_found"]
        elif has_compl:
            baseline_types = ["complaint"]
        elif "?" in low:
            baseline_types = ["info"]
        else:
            baseline_types = list(active.keys()) if active and ms <= 1 else ["other"]

    # 7) LLM gating
    llm_result = None
    if settings.LLM_ENABLED and settings.OPENAI_API_KEY and ms >= 2 and not isinstance(sess.get("pending"), dict):
        # context is minimal and hashed: sharedSlots + short summaries
        context = {
            "sharedSlots": sess.get("sharedSlots") if isinstance(sess.get("sharedSlots"), dict) else {},
            "activeCaseTypes": list(active.keys()),
            "activeExtracted": {k: (v.get("extracted") or {}) for k, v in active.items()},
        }
        llm_result = await llm_extract(chat_id=chat_id, user_text=t, context=context)

    llm_case_types: List[str] = []
    llm_slots_patch: Dict[str, Any] = {}
    llm_categories: Optional[List[str]] = None
    llm_severity: Optional[int] = None
    llm_question: Optional[str] = None
    llm_ask_slots: List[str] = []
    llm_bundle: bool = False
    llm_tone: Optional[str] = None

    if llm_result:
        # intents by threshold
        thr = settings.LLM_CONFIDENCE_THRESHOLD
        intents_sorted = sorted(llm_result.intents, key=lambda x: x.confidence_0_100, reverse=True)
        for it in intents_sorted:
            if it.confidence_0_100 >= thr and it.type in ("complaint", "lost_and_found", "gratitude", "info"):
                if it.type not in llm_case_types:
                    llm_case_types.append(it.type)

        s = sanitize_slots(llm_result.slots)
        llm_slots_patch = s.model_dump(exclude_none=True)

        llm_categories = llm_result.complaint_meta.categories
        llm_severity = llm_result.complaint_meta.severity_1_5

        llm_tone = llm_result.tone
        if llm_result.next_action and llm_result.next_action.kind == "ask":
            llm_question = llm_result.next_action.question
            llm_ask_slots = list(llm_result.next_action.ask_slots or [])
            llm_bundle = bool(llm_result.next_action.bundle)

        # hard override stays: gratitude-only
        if has_grat and not has_lost and not has_compl:
            llm_case_types = ["gratitude"]

    # 8) resolve final case types
    if target_case_types:
        final_case_types = target_case_types
    else:
        if llm_case_types:
            final_case_types = llm_case_types
        else:
            final_case_types = baseline_types

    if final_case_types == ["other"]:
        await send_text(channel_id, chat_type, chat_id, "Уточните: это жалоба, благодарность или забытая вещь?", f"bot-clarify-{message_id}")
        return

    # 9) ensure/create cases
    cases: Dict[str, Dict[str, Any]] = {}
    for ct in final_case_types:
        if ct not in ("complaint", "lost_and_found", "gratitude", "info"):
            continue
        cases[ct] = await get_or_create_case(m, sess, channel_id, chat_id, chat_type, contact_name, ct)

    # 10) merge shared slots (train/car) from rule+LLM
    shared_patch = {}
    if llm_slots_patch.get("train"):
        shared_patch["train"] = llm_slots_patch["train"]
    if llm_slots_patch.get("carNumber") is not None:
        shared_patch["carNumber"] = llm_slots_patch["carNumber"]

    # rule extraction also
    rule_ents = extract_entities_rule(t)
    if rule_ents.get("train"):
        shared_patch["train"] = rule_ents["train"]
    if rule_ents.get("carNumber") is not None:
        shared_patch["carNumber"] = rule_ents["carNumber"]

    if shared_patch:
        await set_shared_slots(m, channel_id, chat_id, shared_patch)

    # refresh session
    sess = await ensure_session(m, channel_id, chat_id, chat_type)

    # 11) update each case
    updated: Dict[str, Dict[str, Any]] = {}
    for ct, c in cases.items():
        # categories+severity only for complaint
        updated[ct] = await update_case(
            m,
            c,
            message_id=message_id,
            text=t,
            msg_type=msg_type,
            content_uri=content_uri,
            sess=sess,
            llm_slots=llm_slots_patch,
            llm_categories=(llm_categories if ct == "complaint" else None),
            llm_severity=(llm_severity if ct == "complaint" else None),
        )

    # refresh session after pending clear
    sess = await ensure_session(m, channel_id, chat_id, chat_type)
    active = await load_active_cases(m, sess)

    # 12) dispatch ready cases
    dispatched_ids: Dict[str, str] = {}

    # complaint dispatch
    if "complaint" in active and case_ready(active["complaint"]):
        c = active["complaint"]
        ex = c.get("extracted") or {}
        region = resolve_region(ex.get("train"), ex.get("routeFrom"), ex.get("routeTo"))
        executor = resolve_executor(region)
        if executor.target_chat_id and executor.target_chat_type:
            await send_text(
                channel_id,
                executor.target_chat_type,
                executor.target_chat_id,
                format_dispatch_text(c) + f"\nРегион: {executor.region}",
                f"dispatch-complaint-{c['caseId']}",
            )
        await m.cases.update_one({"caseId": c["caseId"]}, {"$set": {"status": "sent", "updatedAt": _now_utc()}})
        dispatched_ids["complaint"] = c["caseId"]

    # lost dispatch
    if "lost_and_found" in active and case_ready(active["lost_and_found"]):
        c = active["lost_and_found"]
        target = getattr(settings, "LOST_FOUND_TARGET", None) or {}
        if target.get("chatId") and target.get("chatType"):
            await send_text(
                channel_id,
                target["chatType"],
                target["chatId"],
                format_dispatch_text(c),
                f"dispatch-lost-{c['caseId']}",
            )
        await m.cases.update_one({"caseId": c["caseId"]}, {"$set": {"status": "sent", "updatedAt": _now_utc()}})
        dispatched_ids["lost_and_found"] = c["caseId"]

    # gratitude “dispatch” (обычно достаточно ack клиенту; при желании можно слать в отдельный чат)
    if "gratitude" in active and case_ready(active["gratitude"]):
        c = active["gratitude"]
        await m.cases.update_one({"caseId": c["caseId"]}, {"$set": {"status": "closed", "updatedAt": _now_utc()}})
        dispatched_ids["gratitude"] = c["caseId"]

    # info “dispatch” (по желанию)
    if "info" in active and case_ready(active["info"]):
        c = active["info"]
        await m.cases.update_one({"caseId": c["caseId"]}, {"$set": {"status": "sent", "updatedAt": _now_utc()}})
        dispatched_ids["info"] = c["caseId"]

    if dispatched_ids:
        await clear_pending(m, channel_id, chat_id)
        await send_text(channel_id, chat_type, chat_id, format_user_ack(dispatched_ids), f"bot-ack-{message_id}")
        return

    # 13) ask next question (one per turn)
    missing_by_type: Dict[str, List[str]] = {}
    for ct, c in active.items():
        miss = required_slots(c)
        if miss:
            missing_by_type[ct] = miss

    if missing_by_type:
        # Если LLM предложил вопрос и он подходит нашим нужным слотам — используем его
        use_llm_question = False
        if llm_question and llm_ask_slots:
            needed = set()
            for miss in missing_by_type.values():
                needed |= set(miss)
            # allow "train_car_bundle" as proxy
            ok = True
            for s in llm_ask_slots:
                if s in ("train_car_bundle", "lost_bundle"):
                    continue
                if s not in needed and s not in ("train", "carNumber", "place", "item", "itemDetails", "when", "complaintText", "gratitudeText", "question"):
                    ok = False
            if ok:
                use_llm_question = True

        if use_llm_question:
            # лёгкая эмпатия для angry, если в вопросе не было
            qq = llm_question.strip()
            if llm_tone == "angry" and not qq.lower().startswith(("понимаю", "извините", "сожалею")):
                qq = "Понимаю вас. " + qq
            await send_text(channel_id, chat_type, chat_id, qq, f"bot-q-llm-{message_id}")
            # pending slots: преобразуем спец-слоты в реальные
            if "train_car_bundle" in llm_ask_slots:
                await set_pending(m, channel_id, chat_id, qq, ["train", "carNumber"], list(missing_by_type.keys()))
            elif "lost_bundle" in llm_ask_slots:
                await set_pending(m, channel_id, chat_id, qq, ["place", "item", "itemDetails", "when"], ["lost_and_found"])
            else:
                await set_pending(m, channel_id, chat_id, qq, llm_ask_slots, list(missing_by_type.keys()))
            return

        # иначе rule-based вопрос
        q, slots, cts = build_question_for(missing_by_type)
        if q and slots:
            await send_text(channel_id, chat_type, chat_id, q, f"bot-q-{message_id}")
            await set_pending(m, channel_id, chat_id, q, slots, cts)
            return

    # fallback
    await send_text(
        channel_id, chat_type, chat_id,
        "Опишите, пожалуйста, поезд, вагон и что произошло (или что потеряли).",
        f"bot-fallback-{message_id}",
    )
