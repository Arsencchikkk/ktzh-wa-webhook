from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from db import MongoStore
from nlu import build_nlu, extract_train_and_car, extract_car, extract_train, normalize


CaseType = str  # "complaint" | "lost_and_found" | "gratitude"


@dataclass
class BotReply:
    text: str
    # for session debug
    asked_slots: List[str]


def _ensure_session_defaults(sess: Dict[str, Any]) -> Dict[str, Any]:
    sess.setdefault("shared", {"train": None, "carNumber": None})
    sess.setdefault("pending", {"slots": [], "bundle": None, "caseTypes": []})
    sess.setdefault("cases", {})  # key: caseType -> caseData
    sess.setdefault("lastBot", {"text": None, "askedSlots": []})
    return sess


def _case_defaults(case_type: CaseType) -> Dict[str, Any]:
    return {
        "type": case_type,
        "status": "collecting",   # collecting | done
        "ticketId": None,
        "slots": {
            "train": None,
            "carNumber": None,
            # complaint
            "complaintText": None,
            # lost
            "place": None,
            "item": None,
            "itemDetails": None,
            "when": None,
            # gratitude
            "staffName": None,
            "gratitudeText": None,
        }
    }


def _required_slots(case_type: CaseType) -> List[str]:
    if case_type == "complaint":
        return ["train", "carNumber", "complaintText"]
    if case_type == "lost_and_found":
        return ["train", "carNumber", "place", "item", "when"]
    if case_type == "gratitude":
        # staffName optional (some people don’t know). But gratitudeText required.
        return ["train", "carNumber", "gratitudeText"]
    return []


def _missing_slots(case: Dict[str, Any]) -> List[str]:
    req = _required_slots(case["type"])
    slots = case["slots"]
    return [k for k in req if not slots.get(k)]


def _apply_shared_into_case(shared: Dict[str, Any], case: Dict[str, Any]) -> None:
    if shared.get("train") and not case["slots"].get("train"):
        case["slots"]["train"] = shared["train"]
    if shared.get("carNumber") and not case["slots"].get("carNumber"):
        case["slots"]["carNumber"] = shared["carNumber"]


def _set_shared(shared: Dict[str, Any], train: Optional[str], car: Optional[int]) -> None:
    if train:
        shared["train"] = train
    if car:
        shared["carNumber"] = car


def _parse_lost_bundle(text: str) -> Dict[str, Optional[str]]:
    """
    Very simple rule parsing:
      place: look for "место", "купе", "тамбур", "полка", etc. else take as free text later.
      item: try to find "сумка/рюкзак/..." else ask.
      when: try to find "сегодня/вчера/..." or time digits; else ask.
    """
    t = normalize(text)

    place = None
    for key in ["место", "купе", "тамбур", "полка", "у окна", "у двери", "багаж", "верхняя", "нижняя"]:
        if key in t:
            place = text.strip()
            break

    item = None
    for key in ["сумк", "рюкзак", "кошел", "телефон", "документ", "паспорт", "карта", "чемодан", "ноутбук", "пакет"]:
        if key in t:
            item = text.strip()
            break

    when = None
    for key in ["сегодня", "вчера", "позавчера", "утром", "вечером", "ночью", "только что"]:
        if key in t:
            when = text.strip()
            break
    # crude: contains time-like digits "12:30" or "18 00"
    import re
    if not when and re.search(r"\b\d{1,2}[:.]\d{2}\b", text) or re.search(r"\b\d{1,2}\s*(час|ч)\b", t):
        when = text.strip()

    return {"place": place, "item": item, "when": when}


def _parse_staff_name(text: str) -> Optional[str]:
    """
    Very light: if user writes 'проводник Иван' / 'Иван' after 'кого' question.
    We'll just return full text; later you can improve.
    """
    tt = text.strip()
    if len(tt) >= 2:
        return tt
    return None


class DialogManager:
    def __init__(self, store: MongoStore) -> None:
        self.store = store

    def handle(self, chat_id_hash: str, chat_meta: Dict[str, Any], user_text: str) -> BotReply:
        sess = self.store.get_session(chat_id_hash) or {}
        sess = _ensure_session_defaults(sess)

        pending_slots: List[str] = sess["pending"].get("slots", [])
        nlu = build_nlu(user_text, pending_slots)

        if nlu.isReset:
            self.store.reset_session(chat_id_hash)
            return BotReply("Ок, сбросил диалог. Напишите, что случилось: жалоба / потерял вещь / благодарность.", [])

        # Always store message
        self.store.add_message({
            "chatIdHash": chat_id_hash,
            "chatId": chat_meta.get("chatId"),
            "channelId": chat_meta.get("channelId"),
            "chatType": chat_meta.get("chatType"),
            "direction": "inbound",
            "text": user_text,
            "dateTime": chat_meta.get("dateTime"),
            "raw": chat_meta.get("raw"),
        })

        # 1) Greeting only => no case creation
        if nlu.isGreetingOnly:
            reply = "Здравствуйте! Напишите, пожалуйста, что именно нужно:\n1) Жалоба\n2) Потерял(а) вещь\n3) Благодарность"
            sess["lastBot"] = {"text": reply, "askedSlots": []}
            sess["pending"] = {"slots": [], "bundle": None, "caseTypes": []}
            self.store.upsert_session(chat_id_hash, sess | {
                "chatId": chat_meta.get("chatId"),
                "channelId": chat_meta.get("channelId"),
                "chatType": chat_meta.get("chatType"),
            })
            return BotReply(reply, [])

        # 2) If message is low-meaning and no pending => ask to clarify (don’t create cases)
        if nlu.meaningScore_0_100 < 30 and not pending_slots:
            reply = "Не совсем понял. Опишите проблему чуть подробнее или напишите: «жалоба», «потерял вещь», «благодарность»."
            sess["lastBot"] = {"text": reply, "askedSlots": []}
            self.store.upsert_session(chat_id_hash, sess | {
                "chatId": chat_meta.get("chatId"),
                "channelId": chat_meta.get("channelId"),
                "chatType": chat_meta.get("chatType"),
            })
            return BotReply(reply, [])

        # 3) Pending-slots handling: treat short answers as slot answers
        if pending_slots:
            filled_any = self._try_fill_pending(sess, user_text)
            if filled_any:
                # after filling, continue normal flow to decide next question / closure
                pass

        # 4) Update shared slots from this message (train/car)
        _set_shared(sess["shared"], nlu.train, nlu.carNumber)

        # 5) Create/activate cases based on intents (multi-intent allowed)
        # Hard rule: if message contains "благодарность/спасибо" => gratitude case must exist
        if nlu.intents:
            for it in nlu.intents:
                if it not in sess["cases"]:
                    sess["cases"][it] = _case_defaults(it)

        # If user wrote complaint/lost/gratitude text, attach to corresponding case
        if "complaint" in sess["cases"] and nlu.complaintText:
            # But avoid storing just "Т58, 7 вагон" as complaintText if that’s all they said:
            # If text contains only train/car and very short => treat it as slot answer only.
            only_train_car = False
            train, car = extract_train_and_car(user_text)
            stripped = user_text.strip()
            if (train or car) and len(stripped) <= 20 and not any(w in normalize(stripped) for w in ["плохо", "опозд", "гряз", "хам", "слом"]):
                only_train_car = True
            if not only_train_car:
                sess["cases"]["complaint"]["slots"]["complaintText"] = user_text.strip()

        if "gratitude" in sess["cases"] and nlu.gratitudeText:
            sess["cases"]["gratitude"]["slots"]["gratitudeText"] = user_text.strip()

        # lost: do not auto-fill place/item/when from the first phrase too aggressively,
        # but we can detect partials.
        if "lost_and_found" in sess["cases"] and nlu.lostHint:
            # Keep hint as itemDetails if user described it
            if len(nlu.lostHint.strip()) >= 10 and not sess["cases"]["lost_and_found"]["slots"].get("itemDetails"):
                sess["cases"]["lost_and_found"]["slots"]["itemDetails"] = nlu.lostHint.strip()

        # 6) Apply shared slots into each active case
        for ct, case in sess["cases"].items():
            _apply_shared_into_case(sess["shared"], case)

        # 7) Decide next question (bundles) OR create tickets if ready
        reply = self._next_step(chat_id_hash, sess)

        # persist session
        sess["chatId"] = chat_meta.get("chatId")
        sess["channelId"] = chat_meta.get("channelId")
        sess["chatType"] = chat_meta.get("chatType")
        sess["lastBot"] = {"text": reply.text, "askedSlots": reply.asked_slots}
        self.store.upsert_session(chat_id_hash, sess)
        return reply

    # ---------------- internals ----------------

    def _try_fill_pending(self, sess: Dict[str, Any], user_text: str) -> bool:
        """
        Fill pending bundle slots from user message.
        Key feature: short "8" fills carNumber if we were waiting for it.
        """
        slots_to_fill: List[str] = sess["pending"].get("slots", [])
        if not slots_to_fill:
            return False

        text = user_text.strip()
        changed = False

        # Try parse train+car from message
        train, car = extract_train_and_car(text)

        # Fill shared if relevant
        if "train" in slots_to_fill and train and not sess["shared"].get("train"):
            sess["shared"]["train"] = train
            changed = True
        if "carNumber" in slots_to_fill and car and not sess["shared"].get("carNumber"):
            sess["shared"]["carNumber"] = car
            changed = True

        # Fill case-specific pending slots
        # Determine which cases this pending question was for
        case_types: List[str] = sess["pending"].get("caseTypes", [])
        bundle = sess["pending"].get("bundle")

        def set_for_cases(key: str, value: Any) -> None:
            nonlocal changed
            for ct in case_types:
                if ct in sess["cases"]:
                    if not sess["cases"][ct]["slots"].get(key) and value:
                        sess["cases"][ct]["slots"][key] = value
                        changed = True

        if bundle == "lost_bundle":
            parsed = _parse_lost_bundle(text)
            if "place" in slots_to_fill and parsed["place"]:
                set_for_cases("place", parsed["place"])
            if "item" in slots_to_fill and parsed["item"]:
                set_for_cases("item", parsed["item"])
            if "when" in slots_to_fill and parsed["when"]:
                set_for_cases("when", parsed["when"])
            # If user gave long description, treat as itemDetails
            if len(text) >= 10:
                set_for_cases("itemDetails", text)

        if bundle == "gratitude_bundle":
            # expecting staffName + gratitudeText maybe
            if "staffName" in slots_to_fill and len(text) >= 2:
                set_for_cases("staffName", _parse_staff_name(text))
            # if they wrote details of gratitude
            if "gratitudeText" in slots_to_fill and len(text) >= 5:
                set_for_cases("gratitudeText", text)

        if bundle == "complaint_text":
            if "complaintText" in slots_to_fill and len(text) >= 5:
                set_for_cases("complaintText", text)

        # If something changed, clear pending to avoid re-asking
        if changed:
            sess["pending"] = {"slots": [], "bundle": None, "caseTypes": []}
        return changed

    def _next_step(self, chat_id_hash: str, sess: Dict[str, Any]) -> BotReply:
        # if no cases yet but message has meaning -> ask what they need
        if not sess["cases"]:
            return BotReply(
                "Ок. Это жалоба, потеря вещи или благодарность? Напишите одним словом: «жалоба» / «потерял вещь» / «благодарность».",
                []
            )

        # 1) First: ensure shared train+car if any open case needs them
        needs_train = any("train" in _missing_slots(case) for case in sess["cases"].values())
        needs_car = any("carNumber" in _missing_slots(case) for case in sess["cases"].values())
        if needs_train or needs_car:
            # Ask bundled once
            sess["pending"] = {
                "slots": ["train", "carNumber"],
                "bundle": "train_car",
                "caseTypes": list(sess["cases"].keys()),
            }
            return BotReply(
                "Уточните, пожалуйста, номер поезда и номер вагона (можно одним сообщением).\n"
                "Пример: Т58, 7 вагон",
                ["train", "carNumber"]
            )

        # 2) Then: for each case, collect missing slots with bundles
        # Priority: lost details first (time-sensitive), then complaint, then gratitude
        order = ["lost_and_found", "complaint", "gratitude"]
        for ct in order:
            if ct not in sess["cases"]:
                continue
            case = sess["cases"][ct]
            miss = _missing_slots(case)
            if not miss:
                continue

            # lost bundle: ask place+item+when together
            if ct == "lost_and_found":
                sess["pending"] = {
                    "slots": ["place", "item", "when"],
                    "bundle": "lost_bundle",
                    "caseTypes": ["lost_and_found"],
                }
                return BotReply(
                    "Понял(а). Чтобы помочь с поиском, напишите одним сообщением:\n"
                    "1) где примерно оставили в вагоне (место/купе/полка/тамбур)\n"
                    "2) что за вещь и приметы (цвет/бренд/что внутри)\n"
                    "3) когда это было (сегодня/вчера/время)\n"
                    "Пример: «7 вагон, место 12 (верхняя полка), черная сумка Nike, вчера в 18:30»",
                    ["place", "item", "when"]
                )

            if ct == "complaint":
                sess["pending"] = {"slots": ["complaintText"], "bundle": "complaint_text", "caseTypes": ["complaint"]}
                return BotReply(
                    "Опишите, пожалуйста, суть жалобы (что произошло). Можно коротко, но по делу.",
                    ["complaintText"]
                )

            if ct == "gratitude":
                # train+car already collected; now ask who and for what
                # gratitudeText often contains only "благодарность" => request details
                if not case["slots"].get("gratitudeText") or normalize(case["slots"]["gratitudeText"]) in {"благодарность", "спасибо", "рахмет"}:
                    sess["pending"] = {
                        "slots": ["staffName", "gratitudeText"],
                        "bundle": "gratitude_bundle",
                        "caseTypes": ["gratitude"],
                    }
                    return BotReply(
                        "Принял(а) благодарность 🙌 Напишите, пожалуйста:\n"
                        "1) кого вы хотите поблагодарить (например: проводник, кассир, имя если знаете)\n"
                        "2) за что именно\n"
                        "Можно одним сообщением.",
                        ["staffName", "gratitudeText"]
                    )

        # 3) If all cases complete -> create tickets
        created_lines = []
        for ct, case in sess["cases"].items():
            miss = _missing_slots(case)
            if miss:
                continue
            if case.get("ticketId"):
                continue

            payload = {
                "type": ct,
                "shared": sess["shared"],
                "slots": case["slots"],
            }
            ticket_id = self.store.create_case(chat_id_hash, ct, payload)
            case["ticketId"] = ticket_id
            case["status"] = "done"
            created_lines.append((ct, ticket_id))

        if created_lines:
            # Clear pending
            sess["pending"] = {"slots": [], "bundle": None, "caseTypes": []}

            # Nice multi-ticket message
            parts = []
            for ct, tid in created_lines:
                label = "Жалоба" if ct == "complaint" else ("Забытые вещи" if ct == "lost_and_found" else "Благодарность")
                parts.append(f"{label}: номер заявки {tid}")

            return BotReply("Принял(а). " + "\n".join(parts), [])

        # 4) Nothing new to ask / already done
        return BotReply("Принял(а). Если хотите добавить детали — просто напишите одним сообщением.", [])
