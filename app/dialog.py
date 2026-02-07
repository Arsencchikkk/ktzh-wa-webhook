from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
import uuid

from .db import MongoStore
from .settings import settings
from . import rules


@dataclass
class BotReply:
    text: str


def new_ticket_id(prefix: str) -> str:
    # KTZH-YYYYMMDD-XXXX
    ts = time.strftime("%Y%m%d", time.gmtime())
    rnd = uuid.uuid4().hex[:8].upper()
    return f"KTZH-{prefix}-{ts}-{rnd}"


class DialogManager:
    def __init__(self, store: MongoStore):
        self.store = store

    async def handle(self, chat_id_hash: str, chat_meta: Dict[str, Any], user_text: str) -> BotReply:
        user_text = (user_text or "").strip()
        now = time.time()

        session = await self.store.get_session(chat_id_hash)
        if not session:
            session = self._new_session(chat_meta)

        session.setdefault("history", [])
        session.setdefault("shared", {"train": None, "car": None})
        session.setdefault("cases", [])
        session.setdefault("pending", None)
        session.setdefault("aggression", 0)
        session.setdefault("flood", {"window_start": now, "count": 0})

        # log user message
        await self.store.log_message(chat_id_hash, "user", user_text, {"chat": chat_meta})
        self._push_history(session, "user", user_text)

        # flood / aggression
        session["aggression"] = max(0, int(session.get("aggression", 0)) - 1)  # легкий decay
        session["aggression"] += rules.detect_aggression(user_text)

        self._update_flood(session, now)
        is_flood = session["flood"]["count"] >= settings.FLOOD_MAX_MSG
        is_angry = session["aggression"] >= 2

        # 0) greeting-only — не создаём кейс
        if rules.is_greeting_only(user_text):
            reply = "Здравствуйте! Напишите, пожалуйста, что случилось: опоздание, забытая вещь или благодарность."
            await self._save(session, chat_id_hash, reply, chat_meta)
            return BotReply(reply)

        # 1) Если pending-slot активен — сначала пытаемся заполнить его коротким ответом
        pending = session.get("pending")
        if pending:
            filled = self._apply_pending(session, user_text)
            if filled:
                # после заполнения — продолжаем диалог по кейсам
                reply = await self._continue_after_update(session, chat_id_hash, chat_meta, is_angry=is_angry, is_flood=is_flood)
                return BotReply(reply)

        # 2) смысл есть/нет
        sc = rules.meaning_score(user_text)
        if sc < settings.MEANING_MIN_SCORE:
            # если нет pending и смысла мало — мягко попросим уточнить
            reply = "Понял. Опишите, пожалуйста, проблему одним сообщением (например: 'поезд T58, вагон 7, оставил сумку')."
            if is_angry or is_flood:
                reply = "Напишите: номер поезда и вагон, и что случилось."
            await self._save(session, chat_id_hash, reply, chat_meta)
            return BotReply(reply)

        # 3) cancel / нашлась вещь / отмена
        if rules.detect_cancel(user_text):
            closed_any = self._close_open_cases(session, reason="cancel_by_user")
            if closed_any:
                reply = "Принято. Я закрыл(а) ваши активные заявки. Если нужно — опишите новую проблему."
                if is_angry or is_flood:
                    reply = "Ок. Закрыл(а) заявки."
                await self._save(session, chat_id_hash, reply, chat_meta)
                return BotReply(reply)

        # 4) детект intents + слоты
        intents = rules.detect_intents(user_text)
        slots = rules.extract_slots(user_text)
        self._merge_shared(session, slots)

        # 5) если intents пустой — считаем как общую "жалобу/сообщение" и уточняем
        if not intents:
            reply = await self._ask_primary_intent(session, chat_id_hash, chat_meta, is_angry=is_angry, is_flood=is_flood)
            return BotReply(reply)

        # 6) создаём кейсы (мульти)
        for it in intents:
            self._ensure_case(session, it)

        # 7) кейс-специфичные поля из текста
        self._apply_case_text(session, intents, user_text)

        # 8) продолжаем: задаём следующий лучший вопрос / или создаём заявки
        reply = await self._continue_after_update(session, chat_id_hash, chat_meta, is_angry=is_angry, is_flood=is_flood)
        return BotReply(reply)

    # ---------------- internal helpers ----------------

    def _new_session(self, chat_meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "version": 1,
            "chat": {
                "chatId": chat_meta.get("chatId"),
                "channelId": chat_meta.get("channelId"),
                "chatType": chat_meta.get("chatType"),
            },
            "history": [],
            "shared": {"train": None, "car": None},
            "cases": [],
            "pending": None,
            "aggression": 0,
            "flood": {"window_start": time.time(), "count": 0},
        }

    def _push_history(self, session: Dict[str, Any], role: str, text: str) -> None:
        h = session["history"]
        h.append({"role": role, "text": text, "ts": time.time()})
        if len(h) > settings.MAX_HISTORY:
            del h[:-settings.MAX_HISTORY]

    def _update_flood(self, session: Dict[str, Any], now: float) -> None:
        fw = session.get("flood") or {"window_start": now, "count": 0}
        ws = float(fw.get("window_start", now))
        if now - ws > settings.FLOOD_WINDOW_SEC:
            fw["window_start"] = now
            fw["count"] = 0
        fw["count"] = int(fw.get("count", 0)) + 1
        session["flood"] = fw

    def _merge_shared(self, session: Dict[str, Any], slots: Dict[str, object]) -> None:
        shared = session["shared"]
        if "train" in slots and slots["train"]:
            shared["train"] = slots["train"]
        if "car" in slots and slots["car"]:
            shared["car"] = slots["car"]

    def _ensure_case(self, session: Dict[str, Any], case_type: str) -> None:
        for c in session["cases"]:
            if c["type"] == case_type and c["status"] in ("draft", "open"):
                return
        session["cases"].append(
            {
                "id": uuid.uuid4().hex,
                "type": case_type,  # complaint|lost|gratitude
                "status": "draft",  # draft -> open -> closed
                "ticket_id": None,
                "slots": {},
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        )

    def _apply_case_text(self, session: Dict[str, Any], intents: List[str], user_text: str) -> None:
        t = user_text.strip()
        for c in session["cases"]:
            if c["type"] not in intents:
                continue
            s = c["slots"]
            if c["type"] == "complaint":
                # complaint_text если это не просто "жалоба" одним словом
                if len(t) >= 6:
                    s.setdefault("complaint_text", t)
            elif c["type"] == "lost":
                if len(t) >= 6:
                    s.setdefault("item_text", t)
            elif c["type"] == "gratitude":
                if len(t) >= 4:
                    s.setdefault("gratitude_text", t)
            c["updated_at"] = time.time()

    def _apply_pending(self, session: Dict[str, Any], user_text: str) -> bool:
        """
        pending = {
          "slots": ["train","car"] or ["car"] ...
          "scope": "shared" | "case:<id>"
        }
        """
        pending = session.get("pending")
        if not pending or not isinstance(pending, dict):
            return False

        needed: List[str] = list(pending.get("slots") or [])
        if not needed:
            session["pending"] = None
            return False

        filled_any = False
        text = user_text

        # пытаемся вытянуть train/car/place по очереди
        if "train" in needed:
            tr = rules.parse_train(text)
            if not tr:
                # если ожидаем поезд и клиент пишет просто "58"
                num = rules.parse_short_number(text)
                if num and 1 <= num <= 9999:
                    tr = f"T{num}"
            if tr:
                session["shared"]["train"] = tr
                needed.remove("train")
                filled_any = True

        if "car" in needed:
            car = rules.parse_car(text)
            if car is None:
                # если ожидаем вагон — короткая цифра ок
                num = rules.parse_short_number(text)
                if num and 1 <= num <= 99:
                    car = num
            if car is not None:
                session["shared"]["car"] = car
                needed.remove("car")
                filled_any = True

        # case-specific pending (lost: place)
        if "place" in needed:
            pl = rules.parse_place(text)
            if pl:
                # scope может быть case:<id>
                scope = pending.get("scope", "")
                if scope.startswith("case:"):
                    cid = scope.split(":", 1)[1]
                    for c in session["cases"]:
                        if c["id"] == cid:
                            c["slots"]["place"] = pl
                            break
                needed.remove("place")
                filled_any = True

        pending["slots"] = needed
        if not needed:
            session["pending"] = None
        else:
            session["pending"] = pending

        return filled_any

    def _close_open_cases(self, session: Dict[str, Any], reason: str) -> bool:
        changed = False
        for c in session["cases"]:
            if c["status"] in ("draft", "open"):
                c["status"] = "closed"
                c["closed_reason"] = reason
                c["updated_at"] = time.time()
                changed = True
        return changed

    async def _ask_primary_intent(self, session: Dict[str, Any], chat_id_hash: str, chat_meta: Dict[str, Any],
                                 is_angry: bool, is_flood: bool) -> str:
        reply = "Уточните, пожалуйста, что именно нужно: жалоба, забытая вещь или благодарность?"
        if is_angry or is_flood:
            reply = "Что именно: жалоба / забытая вещь / благодарность?"
        await self._save(session, chat_id_hash, reply, chat_meta)
        return reply

    def _required_slots_for_case(self, case_type: str) -> List[str]:
        # shared: train, car
        if case_type == "complaint":
            return ["train", "car", "complaint_text"]
        if case_type == "lost":
            # в lost обязательно ещё место+описание
            return ["train", "car", "place", "item_text"]
        if case_type == "gratitude":
            # благодарность — не закрывать сразу: спрашиваем кого + за что + поезд/вагон
            return ["train", "car", "gratitude_text"]
        return ["train", "car"]

    def _case_missing(self, session: Dict[str, Any], case: Dict[str, Any]) -> List[str]:
        shared = session["shared"]
        slots = case["slots"]
        missing: List[str] = []
        for rs in self._required_slots_for_case(case["type"]):
            if rs in ("train", "car"):
                if not shared.get(rs):
                    missing.append(rs)
            else:
                if not slots.get(rs):
                    missing.append(rs)
        return missing

    def _next_question_for_missing(self, case_type: str, missing: List[str], is_angry: bool) -> str:
        # bundle train+car вместе
        if "train" in missing or "car" in missing:
            if is_angry:
                return "Напишите номер поезда и вагон (например: T58, вагон 7)."
            return "Уточните номер поезда и номер вагона (можно одним сообщением, например: T58, вагон 7)."

        if case_type == "complaint":
            if "complaint_text" in missing:
                return "Опишите, пожалуйста, что произошло (например: 'поезд задержался на 1 час')."

        if case_type == "lost":
            # bundle по lost
            if "place" in missing and "item_text" in missing:
                return "Где примерно в вагоне оставили и что это за вещь? (место/купе/полка + описание)."
            if "place" in missing:
                return "Уточните место в вагоне (место/купе/полка/тамбур)."
            if "item_text" in missing:
                return "Опишите, пожалуйста, вещь (что именно, цвет, приметы)."

        if case_type == "gratitude":
            if "gratitude_text" in missing:
                return "Кого хотите поблагодарить и за что? (например: 'проводника, помог с багажом')."

        return "Уточните, пожалуйста, детали."

    async def _continue_after_update(self, session: Dict[str, Any], chat_id_hash: str, chat_meta: Dict[str, Any],
                                    is_angry: bool, is_flood: bool) -> str:
        # 1) если есть draft/open кейсы — выбираем тот, где больше всего не хватает
        draft_cases = [c for c in session["cases"] if c["status"] in ("draft", "open")]
        if not draft_cases:
            reply = "Понял(а). Чем ещё могу помочь?"
            if is_angry or is_flood:
                reply = "Ок. Что ещё?"
            await self._save(session, chat_id_hash, reply, chat_meta)
            return reply

        # приоритет: lost -> complaint -> gratitude (чтобы быстрее спасать вещи)
        priority = {"lost": 0, "complaint": 1, "gratitude": 2}
        draft_cases.sort(key=lambda c: (priority.get(c["type"], 9), len(self._case_missing(session, c))))

        # 2) если есть кейс полностью заполненный — “создаём заявку” (ticket_id)
        for c in draft_cases:
            missing = self._case_missing(session, c)
            if not missing and not c.get("ticket_id"):
                prefix = {"complaint": "CMP", "lost": "LAF", "gratitude": "THX"}.get(c["type"], "GEN")
                c["ticket_id"] = new_ticket_id(prefix)
                c["status"] = "open"
                c["updated_at"] = time.time()

        # 3) если после создания tickets осталось что-то спрашивать — задаём ОДИН лучший вопрос
        for c in draft_cases:
            missing = self._case_missing(session, c)
            if missing:
                # если missing train/car — ставим pending shared bundle
                if "train" in missing or "car" in missing:
                    session["pending"] = {"slots": ["train", "car"], "scope": "shared"}
                else:
                    # case-specific pending (например place)
                    if "place" in missing:
                        session["pending"] = {"slots": ["place"], "scope": f"case:{c['id']}"}

                reply = self._next_question_for_missing(c["type"], missing, is_angry=is_angry or is_flood)
                await self._save(session, chat_id_hash, reply, chat_meta)
                return reply

        # 4) если всё открыто — подтверждаем пользователю список заявок
        opened = [c for c in session["cases"] if c["status"] == "open" and c.get("ticket_id")]
        if opened:
            ids = ", ".join(c["ticket_id"] for c in opened)
            reply = f"Принял(а). Ваши заявки зарегистрированы: {ids}."
            if is_angry or is_flood:
                reply = f"Ок. Заявки: {ids}."
            await self._save(session, chat_id_hash, reply, chat_meta)
            return reply

        reply = "Понял(а). Уточните, пожалуйста, детали."
        if is_angry or is_flood:
            reply = "Уточните детали."
        await self._save(session, chat_id_hash, reply, chat_meta)
        return reply

    async def _save(self, session: Dict[str, Any], chat_id_hash: str, bot_text: str, chat_meta: Dict[str, Any]) -> None:
        await self.store.log_message(chat_id_hash, "bot", bot_text, {"chat": chat_meta})
        self._push_history(session, "bot", bot_text)
        await self.store.save_session(chat_id_hash, session)
