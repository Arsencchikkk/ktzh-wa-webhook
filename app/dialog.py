from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .db import MongoStore
from .nlu import build_nlu, extract_train_and_car, detect_aggression_and_flood, normalize



CaseType = str  # complaint | lost_and_found | gratitude


@dataclass
class BotReply:
    text: str
    asked_slots: List[str]


def _ensure_session_defaults(sess: Dict[str, Any]) -> Dict[str, Any]:
    sess.setdefault("shared", {"train": None, "carNumber": None})
    sess.setdefault("pending", {"slots": [], "bundle": None, "caseTypes": []})
    sess.setdefault("cases", {})
    sess.setdefault("lastBot", {"text": None, "askedSlots": []})
    sess.setdefault("moderation", {"tone": "neutral", "angry": False, "flooding": False, "repeat_count": 0, "prev_text": None})
    return sess


def _case_defaults(case_type: CaseType) -> Dict[str, Any]:
    return {
        "type": case_type,
        "status": "collecting",
        "ticketId": None,
        "slots": {
            "train": None,
            "carNumber": None,
            "complaintText": None,
            "place": None,
            "item": None,
            "itemDetails": None,
            "when": None,
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

    import re
    if not when and (re.search(r"\b\d{1,2}[:.]\d{2}\b", text) or re.search(r"\b\d{1,2}\s*(час|ч)\b", t)):
        when = text.strip()

    return {"place": place, "item": item, "when": when}


class DialogManager:
    def __init__(self, store: MongoStore) -> None:
        self.store = store

    def handle(self, chat_id_hash: str, chat_meta: Dict[str, Any], user_text: str) -> BotReply:
        sess = _ensure_session_defaults(self.store.get_session(chat_id_hash) or {})
        pending_slots: List[str] = sess["pending"].get("slots", [])

        # --- moderation update (anti-flood / aggression) ---
        mod = detect_aggression_and_flood(
            user_text,
            prev_text=sess["moderation"].get("prev_text"),
            repeat_count=int(sess["moderation"].get("repeat_count") or 0),
        )
        sess["moderation"].update(mod)

        nlu = build_nlu(user_text, pending_slots)

        # store inbound message
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

        # --- auto close: cancel/found ---
        if nlu.isCancel or nlu.isFound:
            if nlu.isCancel:
                closed = self.store.close_open_cases(chat_id_hash, None)
                # also clear local session cases
                sess["cases"] = {}
                sess["pending"] = {"slots": [], "bundle": None, "caseTypes": []}
                txt = "Ок, отменил(а) заявки." if closed > 0 else "Ок, понял(а)."
                if sess["moderation"].get("angry"):
                    txt = "Ок. Закрыл(а)." if closed > 0 else "Ок."
                self._persist(sess, chat_id_hash, chat_meta, txt, [])
                return BotReply(txt, [])

            # found
            closed = self.store.close_open_cases(chat_id_hash, "lost_and_found")
            if "lost_and_found" in sess["cases"]:
                sess["cases"].pop("lost_and_found", None)
            sess["pending"] = {"slots": [], "bundle": None, "caseTypes": []}
            txt = "Отлично! Закрыл(а) заявку по забытым вещам." if closed > 0 else "Отлично! Тогда заявку по забытым вещам не создаю."
            if sess["moderation"].get("angry"):
                txt = "Ок. Закрыл(а) по вещам." if closed > 0 else "Ок."
            self._persist(sess, chat_id_hash, chat_meta, txt, [])
            return BotReply(txt, [])

        # reset
        if nlu.isReset:
            self.store.reset_session(chat_id_hash)
            return BotReply("Ок, сбросил диалог. Напишите: жалоба / потерял вещь / благодарность.", [])

        # greeting only (no case)
        if nlu.isGreetingOnly:
            txt = "Здравствуйте! Напишите: жалоба / потерял вещь / благодарность."
            if sess["moderation"].get("angry"):
                txt = "Напишите: жалоба / потерял / благодарность."
            self._persist(sess, chat_id_hash, chat_meta, txt, [])
            return BotReply(txt, [])

        # low meaning and no pending => clarify (но без спора)
        if nlu.meaningScore_0_100 < 30 and not pending_slots:
            txt = "Не понял. Напишите: жалоба / потерял вещь / благодарность."
            if sess["moderation"].get("angry"):
                txt = "Коротко: жалоба / потерял / благодарность."
            self._persist(sess, chat_id_hash, chat_meta, txt, [])
            return BotReply(txt, [])

        # pending slots fill
        if pending_slots:
            if self._try_fill_pending(sess, user_text):
                pass

        # update shared from msg
        train, car = extract_train_and_car(user_text)
        _set_shared(sess["shared"], train, car)

        # activate cases by intents (multi-intent)
        for it in nlu.intents:
            if it not in sess["cases"]:
                sess["cases"][it] = _case_defaults(it)

        # attach texts
        if "complaint" in sess["cases"] and nlu.complaintText:
            sess["cases"]["complaint"]["slots"]["complaintText"] = user_text.strip()
        if "gratitude" in sess["cases"] and nlu.gratitudeText:
            sess["cases"]["gratitude"]["slots"]["gratitudeText"] = user_text.strip()
        if "lost_and_found" in sess["cases"] and nlu.lostHint:
            if len(nlu.lostHint.strip()) >= 10 and not sess["cases"]["lost_and_found"]["slots"].get("itemDetails"):
                sess["cases"]["lost_and_found"]["slots"]["itemDetails"] = nlu.lostHint.strip()

        # apply shared
        for case in sess["cases"].values():
            _apply_shared_into_case(sess["shared"], case)

        reply = self._next_step(chat_id_hash, sess)

        self._persist(sess, chat_id_hash, chat_meta, reply.text, reply.asked_slots)
        return reply

    def _persist(self, sess: Dict[str, Any], chat_id_hash: str, chat_meta: Dict[str, Any], txt: str, asked: List[str]) -> None:
        sess["chatId"] = chat_meta.get("chatId")
        sess["channelId"] = chat_meta.get("channelId")
        sess["chatType"] = chat_meta.get("chatType")
        sess["lastBot"] = {"text": txt, "askedSlots": asked}
        self.store.upsert_session(chat_id_hash, sess)

    def _try_fill_pending(self, sess: Dict[str, Any], user_text: str) -> bool:
        slots_to_fill: List[str] = sess["pending"].get("slots", [])
        if not slots_to_fill:
            return False

        text = user_text.strip()
        changed = False

        train, car = extract_train_and_car(text)
        if "train" in slots_to_fill and train and not sess["shared"].get("train"):
            sess["shared"]["train"] = train
            changed = True
        if "carNumber" in slots_to_fill and car and not sess["shared"].get("carNumber"):
            sess["shared"]["carNumber"] = car
            changed = True

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
            if len(text) >= 10:
                set_for_cases("itemDetails", text)

        if bundle == "complaint_text":
            if "complaintText" in slots_to_fill and len(text) >= 5:
                set_for_cases("complaintText", text)

        if bundle == "gratitude_bundle":
            if "gratitudeText" in slots_to_fill and len(text) >= 5:
                set_for_cases("gratitudeText", text)

        if changed:
            sess["pending"] = {"slots": [], "bundle": None, "caseTypes": []}
        return changed

    def _next_step(self, chat_id_hash: str, sess: Dict[str, Any]) -> BotReply:
        angry = bool(sess["moderation"].get("angry"))

        if not sess["cases"]:
            txt = "Это жалоба, потеря вещи или благодарность?"
            if angry:
                txt = "Жалоба / потерял / благодарность?"
            return BotReply(txt, [])

        needs_train = any("train" in _missing_slots(case) for case in sess["cases"].values())
        needs_car = any("carNumber" in _missing_slots(case) for case in sess["cases"].values())
        if needs_train or needs_car:
            sess["pending"] = {"slots": ["train", "carNumber"], "bundle": "train_car", "caseTypes": list(sess["cases"].keys())}
            txt = "Укажите поезд и вагон. Пример: Т58, 7."
            if not angry:
                txt = "Уточните номер поезда и номер вагона (можно одним сообщением). Пример: Т58, 7 вагон"
            return BotReply(txt, ["train", "carNumber"])

        # priority: lost -> complaint -> gratitude
        for ct in ["lost_and_found", "complaint", "gratitude"]:
            if ct not in sess["cases"]:
                continue
            case = sess["cases"][ct]
            miss = _missing_slots(case)
            if not miss:
                continue

            if ct == "lost_and_found":
                sess["pending"] = {"slots": ["place", "item", "when"], "bundle": "lost_bundle", "caseTypes": ["lost_and_found"]}
                txt = "Где? Что за вещь? Когда?"
                if not angry:
                    txt = (
                        "Напишите одним сообщением:\n"
                        "1) где в вагоне (место/купе/полка/тамбур)\n"
                        "2) что за вещь и приметы\n"
                        "3) когда (сегодня/вчера/время)"
                    )
                return BotReply(txt, ["place", "item", "when"])

            if ct == "complaint":
                sess["pending"] = {"slots": ["complaintText"], "bundle": "complaint_text", "caseTypes": ["complaint"]}
                txt = "Что произошло? (коротко)"
                if not angry:
                    txt = "Опишите, пожалуйста, суть жалобы (что произошло)."
                return BotReply(txt, ["complaintText"])

            if ct == "gratitude":
                # не закрываем — тянем детали
                sess["pending"] = {"slots": ["gratitudeText"], "bundle": "gratitude_bundle", "caseTypes": ["gratitude"]}
                txt = "Кого благодарите и за что?"
                if not angry:
                    txt = "Принял(а) благодарность 🙌 Кого вы хотите поблагодарить и за что? (можно одним сообщением)"
                return BotReply(txt, ["gratitudeText"])

        # create tickets when ready
        created = []
        for ct, case in sess["cases"].items():
            if _missing_slots(case):
                continue
            if case.get("ticketId"):
                continue
            payload = {"type": ct, "shared": sess["shared"], "slots": case["slots"]}
            ticket_id = self.store.create_case(chat_id_hash, ct, payload)
            case["ticketId"] = ticket_id
            case["status"] = "done"
            created.append((ct, ticket_id))

        if created:
            sess["pending"] = {"slots": [], "bundle": None, "caseTypes": []}
            if angry:
                # максимально коротко
                lines = [f"{tid}" for _, tid in created]
                return BotReply("Принял(а). Заявки: " + ", ".join(lines), [])
            parts = []
            for ct, tid in created:
                label = "Жалоба" if ct == "complaint" else ("Забытые вещи" if ct == "lost_and_found" else "Благодарность")
                parts.append(f"{label}: {tid}")
            return BotReply("Принял(а).\n" + "\n".join(parts), [])

        return BotReply("Принял(а).", [])
