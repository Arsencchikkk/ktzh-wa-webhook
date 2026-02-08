from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import datetime as dt
import re
import secrets

from .nlu import build_nlu, extract_train_and_car, detect_aggression_and_flood, normalize


@dataclass
class BotReply:
    text: str
    meta: Dict[str, Any] | None = None


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _gen_case_id(prefix: str, chat_id_hash: str) -> str:
    """
    ВАЖНО: не детерминированный id, чтобы не было дублей в один день.
    """
    d = _now_utc().strftime("%Y%m%d")
    short_chat = chat_id_hash[:6].upper()
    rnd = secrets.token_hex(3).upper()  # 6 hex chars
    return f"{prefix}-{d}-{short_chat}-{rnd}"


def _short(text: str) -> str:
    return (text or "").strip()


def _is_only_number(text: str) -> Optional[int]:
    t = normalize(text)
    m = re.fullmatch(r"\s*(\d{1,2})\s*", t)
    if not m:
        return None
    v = int(m.group(1))
    return v if 1 <= v <= 99 else None


def _extract_train_fallback(text: str) -> Optional[str]:
    """
    Фолбек для поездов типа:
      - 81/82
      - 10ЦА
      - 123А
    (без префикса Т)
    """
    t = (text or "").strip()
    tn = normalize(t)

    m = re.search(r"\b(\d{1,3}\s*/\s*\d{1,3})\b", tn)
    if m:
        return m.group(1).replace(" ", "").upper()

    m = re.search(r"\b(\d{1,4}\s*[a-zа-я]{1,3})\b", tn)
    if m:
        return m.group(1).replace(" ", "").upper()

    m = re.search(r"\bпоезд\s*(\d{1,3}(?:\s*/\s*\d{1,3})?)\b", tn)
    if m:
        return m.group(1).replace(" ", "").upper()

    return None


def _extract_train_car_any(text: str) -> tuple[Optional[str], Optional[int]]:
    tr, car = extract_train_and_car(text)
    if not tr:
        tr = _extract_train_fallback(text)
    return tr, car


def _is_train_car_only(text: str) -> bool:
    t = normalize(text)
    tr, car = _extract_train_car_any(t)
    tokens = re.findall(r"[a-zа-я0-9/]+", t)
    if not tokens:
        return False

    allowed = {"т", "t", "вагон", "поезд"}
    if tr:
        allowed.add(normalize(tr))
        allowed.add(normalize(tr).replace("т", "").strip())
    if car is not None:
        allowed.add(str(car))

    meaningful = [x for x in tokens if x not in allowed]
    return len(meaningful) <= 1 and len(tokens) <= 6


def _is_generic_complaint(text: str) -> bool:
    t = normalize(text)
    generic = (
        "хочу пожаловаться",
        "хочу жалобу",
        "хочу оставить жалобу",
        "у меня жалоба",
        "жалоба",
        "пожаловаться",
    )
    return any(g in t for g in generic) and len(t.split()) <= 6


def _is_generic_gratitude(text: str) -> bool:
    t = normalize(text)
    generic = ("хочу поблагодарить", "хочу сказать спасибо", "у меня благодарность", "благодарность", "спасибо")
    return any(g in t for g in generic) and len(t.split()) <= 6


def _extract_place(text: str) -> Optional[str]:
    t = normalize(text)

    m = re.search(r"\bместо\s*(\d{1,2})\b|\b(\d{1,2})\s*место\b", t)
    if m:
        num = next((g for g in m.groups() if g and g.isdigit()), None)
        return f"место {num}" if num else None

    m = re.search(r"\bкупе\s*(\d{1,2})\b", t)
    if m:
        return f"купе {m.group(1)}"

    if "тамбур" in t:
        return "тамбур"

    if "верх" in t and "полк" in t:
        return "верхняя полка"
    if "ниж" in t and "полк" in t:
        return "нижняя полка"

    return None


def _extract_when(text: str) -> Optional[str]:
    t = normalize(text)
    if "сегодня" in t:
        return "сегодня"
    if "вчера" in t:
        return "вчера"
    if "позавчера" in t:
        return "позавчера"

    m = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{2,4}))?\b", t)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{d.zfill(2)}.{mo.zfill(2)}.{y}" if y else f"{d.zfill(2)}.{mo.zfill(2)}"

    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        return f"{m.group(1).zfill(2)}:{m.group(2)}"

    return None


def _extract_item(text: str) -> Optional[str]:
    t = normalize(text)
    keys = [
        "сумк", "рюкзак", "чемодан", "пакет",
        "телефон", "документ", "паспорт",
        "кошелек", "бумажник", "наушник", "ноутбук",
    ]
    return _short(text) if any(k in t for k in keys) else None


def _required_slots(case_type: str) -> List[str]:
    if case_type == "lost":
        return ["train", "car", "place", "item", "when"]
    if case_type == "complaint":
        return ["train", "car", "complaintText"]
    if case_type == "gratitude":
        return ["train", "car", "gratitudeText"]
    return []


def _case_title(case_type: str) -> str:
    return {
        "lost": "Забытые/потерянные вещи",
        "complaint": "Жалоба",
        "gratitude": "Благодарность",
    }.get(case_type, case_type)


class DialogManager:
    def __init__(self, store: Any):
        self.store = store
        self.nlu = build_nlu()

    async def _load_session(self, chat_id_hash: str) -> Dict[str, Any]:
        s = None
        if hasattr(self.store, "get_session"):
            s = await self.store.get_session(chat_id_hash)

        if not s:
            s = {
                "shared": {"train": None, "car": None},
                "cases": [],
                "pending": None,
                "moderation": {"prev_text": None, "repeat_count": 0, "last_ts": 0.0},
                # ✅ анти-цикл
                "loop": {"key": None, "count": 0},
                "createdAt": _now_utc().isoformat(),
                "updatedAt": _now_utc().isoformat(),
            }
        # если старая сессия без loop
        if "loop" not in s:
            s["loop"] = {"key": None, "count": 0}
        return s

    async def _save_session(self, chat_id_hash: str, session: Dict[str, Any]) -> None:
        session["updatedAt"] = _now_utc().isoformat()
        if hasattr(self.store, "save_session"):
            await self.store.save_session(chat_id_hash, session)

    # =========================
    # ✅ Anti-loop helpers
    # =========================
    def _loop_bump(self, session: Dict[str, Any], key: str) -> int:
        loop = session.get("loop") or {"key": None, "count": 0}
        if loop.get("key") == key:
            loop["count"] = int(loop.get("count", 0)) + 1
        else:
            loop["key"] = key
            loop["count"] = 1
        session["loop"] = loop
        return int(loop["count"])

    def _loop_reset(self, session: Dict[str, Any]) -> None:
        session["loop"] = {"key": None, "count": 0}

    def _ops_template(self, case_type: str) -> str:
        base = (
            "Похоже, я не могу корректно оформить заявку автоматически.\n"
            "Пожалуйста, отправьте одним сообщением для оперативников по шаблону:\n\n"
            "1) Тип обращения: ЖАЛОБА / ПОТЕРЯЛ(А) ВЕЩЬ / БЛАГОДАРНОСТЬ\n"
            "2) Поезд № (например: Т78 или 10ЦА или 81/82):\n"
            "3) Маршрут (откуда–куда):\n"
            "4) Дата поездки (дд.мм.гггг):\n"
            "5) Время/примерно когда:\n"
            "6) Вагон №:\n"
            "7) Место/купе (если есть):\n"
            "8) Детали обращения (2–4 предложения):\n"
        )
        if case_type == "lost":
            return base.replace("8) Детали обращения", "8) Что потеряли + приметы (цвет/марка) и где оставили")
        return base

    # =========================

    def _get_or_create_case(self, session: Dict[str, Any], case_type: str) -> Dict[str, Any]:
        for c in session["cases"]:
            if c["type"] == case_type and c["status"] in ("open", "collecting"):
                return c

        c = {
            "type": case_type,
            "status": "collecting",
            "slots": {
                "place": None,
                "item": None,
                "when": None,
                "complaintText": None,
                "gratitudeText": None,
                "staffName": None,
            },
            "caseId": None,
            "createdAt": _now_utc().isoformat(),
        }
        session["cases"].append(c)
        return c

    def _reset_dialog(self, session: Dict[str, Any]) -> None:
        session["shared"] = {"train": None, "car": None}
        session["cases"] = []
        session["pending"] = None
        self._loop_reset(session)

    def _close_all_cases(self, session: Dict[str, Any], reason: str) -> None:
        for c in session["cases"]:
            if c["status"] in ("open", "collecting"):
                c["status"] = "closed"
                c["closeReason"] = reason
                c["closedAt"] = _now_utc().isoformat()
        session["pending"] = None
        self._loop_reset(session)

    def _set_pending(self, session: Dict[str, Any], scope: str, slots: List[str], case_type: Optional[str] = None) -> None:
        session["pending"] = {"scope": scope, "slots": slots, "caseType": case_type}

    def _train_car_question_for(self, case_type: str) -> str:
        if case_type == "complaint":
            return "Чтобы оформить жалобу, напишите номер поезда и вагон одним сообщением (пример: Т58, 7 вагон)."
        if case_type == "gratitude":
            return "Чтобы оформить благодарность, напишите номер поезда и вагон (если знаете) одним сообщением (пример: Т58, 7 вагон)."
        if case_type == "lost":
            return "Чтобы помочь найти вещь, напишите номер поезда и вагон одним сообщением (пример: Т58, 7 вагон)."
        return "Уточните номер поезда и номер вагона одним сообщением (пример: Т58, 7 вагон)."

    def _lost_bundle_question(self, angry: bool = False) -> str:
        return (
            "Где в вагоне, что именно и когда оставили? Пример: место 12, черная сумка, вчера 14:30."
            if angry
            else "Для поиска вещи напишите одним сообщением: 1) где в вагоне (место/купе/полка/тамбур), 2) что за вещь и приметы, 3) когда примерно оставили."
        )

    def _is_case_ready(self, session: Dict[str, Any], case: Dict[str, Any]) -> bool:
        shared = session["shared"]
        cs = case["slots"]
        for s in _required_slots(case["type"]):
            if s == "train" and not shared.get("train"):
                return False
            if s == "car" and not shared.get("car"):
                return False
            if s in cs and not cs.get(s):
                return False
        return True

    async def _submit_case(self, chat_id_hash: str, session: Dict[str, Any], case: Dict[str, Any]) -> str:
        case_id = _gen_case_id("KTZH", chat_id_hash)
        case["caseId"] = case_id
        case["status"] = "open"
        case["openedAt"] = _now_utc().isoformat()

        if hasattr(self.store, "create_case"):
            await self.store.create_case({
                "caseId": case_id,          # ✅ обязательно
                "ticketId": case_id,
                "chatIdHash": chat_id_hash,
                "type": case["type"],
                "status": "open",
                "payload": {"shared": session.get("shared"), "slots": case.get("slots")},
            })

        self._loop_reset(session)
        return case_id

    def _apply_pending(self, session: Dict[str, Any], text: str) -> None:
        p = session.get("pending")
        if not p:
            return

        scope = p.get("scope")
        slots: List[str] = p.get("slots") or []
        case_type = p.get("caseType")

        train, car = _extract_train_car_any(text)
        shared = session["shared"]

        if scope == "shared":
            if "train" in slots and train and not shared.get("train"):
                shared["train"] = train
            if "car" in slots:
                if car is not None and not shared.get("car"):
                    shared["car"] = car
                else:
                    n = _is_only_number(text)
                    if n is not None and not shared.get("car"):
                        shared["car"] = n

            ok = True
            if "train" in slots and not shared.get("train"):
                ok = False
            if "car" in slots and not shared.get("car"):
                ok = False
            if ok:
                session["pending"] = None

        if scope == "case" and case_type:
            case = self._get_or_create_case(session, case_type)
            cs = case["slots"]

            if "place" in slots and not cs.get("place"):
                pl = _extract_place(text)
                if pl:
                    cs["place"] = pl

            if "when" in slots and not cs.get("when"):
                wh = _extract_when(text)
                if wh:
                    cs["when"] = wh

            if "item" in slots and not cs.get("item"):
                it = _extract_item(text)
                if it:
                    cs["item"] = it

            if "complaintText" in slots and not cs.get("complaintText"):
                # ✅ если текст длиннее и не только поезд/вагон — считаем деталями
                if (not _is_train_car_only(text)) and (not _is_generic_complaint(text)):
                    cs["complaintText"] = _short(text)

            if "gratitudeText" in slots and not cs.get("gratitudeText"):
                if (not _is_train_car_only(text)) and (not _is_generic_gratitude(text)):
                    cs["gratitudeText"] = _short(text)

            ok = True
            for sname in slots:
                if sname == "train":
                    ok = ok and bool(session["shared"].get("train"))
                elif sname == "car":
                    ok = ok and bool(session["shared"].get("car"))
                else:
                    ok = ok and bool(cs.get(sname))
            if ok:
                session["pending"] = None

    async def handle(self, chat_id_hash: str, chat_meta: Dict[str, Any], user_text: str) -> BotReply:
        session = await self._load_session(chat_id_hash)

        # ✅ FIX: если есть старый unique индекс по (channelId, chatId)
        session["chatId"] = str(chat_meta.get("chatId") or session.get("chatId") or "")
        session["channelId"] = str(chat_meta.get("channelId") or session.get("channelId") or "")
        session["chatType"] = str(chat_meta.get("chatType") or session.get("chatType") or "")

        text = user_text or ""
        tnorm = normalize(text)

        # cancel
        if tnorm in {"стоп", "хватит", "отмена", "прекрати", "прекратите"}:
            self._close_all_cases(session, reason="user_cancel")
            self._reset_dialog(session)
            await self._save_session(chat_id_hash, session)
            return BotReply(text="Ок, остановил. Начнём заново — напишите, что случилось.")

        session, is_angry, is_flood = detect_aggression_and_flood(session, text)
        nlu_res = self.nlu.analyze(text)

        if getattr(nlu_res, "cancel", False):
            self._close_all_cases(session, reason="user_cancel")
            self._reset_dialog(session)
            await self._save_session(chat_id_hash, session)
            return BotReply(text="Ок, остановил. Начнём заново — напишите, что случилось.")

        # если был pending — пробуем применить
        if session.get("pending"):
            self._apply_pending(session, text)

        # greeting only
        if getattr(nlu_res, "greeting_only", False) and not session.get("cases"):
            await self._save_session(chat_id_hash, session)
            return BotReply(text="Здравствуйте! Опишите проблему одним сообщением (опоздание / забытая вещь / жалоба / благодарность).")

        intents: List[str] = list(getattr(nlu_res, "intents", []) or [])

        # shared train/car
        shared = session["shared"]
        slots = getattr(nlu_res, "slots", {}) or {}

        if slots.get("train") and not shared.get("train"):
            shared["train"] = slots["train"]
        if slots.get("car") and not shared.get("car"):
            shared["car"] = slots["car"]

        tr, car = _extract_train_car_any(text)
        if tr and not shared.get("train"):
            shared["train"] = tr
        if car is not None and not shared.get("car"):
            shared["car"] = car

        # create cases
        for it in intents:
            self._get_or_create_case(session, it)

        # fill case texts
        if "complaint" in intents:
            ccase = self._get_or_create_case(session, "complaint")
            if not ccase["slots"].get("complaintText"):
                if (not _is_train_car_only(text)) and (not _is_generic_complaint(text)):
                    ccase["slots"]["complaintText"] = _short(text)

        if "gratitude" in intents:
            gcase = self._get_or_create_case(session, "gratitude")
            if not gcase["slots"].get("gratitudeText"):
                if (not _is_train_car_only(text)) and (not _is_generic_gratitude(text)):
                    gcase["slots"]["gratitudeText"] = _short(text)
            if slots.get("staffName") and not gcase["slots"].get("staffName"):
                gcase["slots"]["staffName"] = str(slots["staffName"])

        if "lost" in intents:
            lcase = self._get_or_create_case(session, "lost")
            if not lcase["slots"].get("place"):
                lcase["slots"]["place"] = _extract_place(text)
            if not lcase["slots"].get("item"):
                lcase["slots"]["item"] = _extract_item(text)
            if not lcase["slots"].get("when"):
                lcase["slots"]["when"] = _extract_when(text)

        # primary case
        primary: Optional[str] = None
        for ct in ["lost", "complaint", "gratitude"]:
            if any(c["type"] == ct and c["status"] in ("open", "collecting") for c in session["cases"]):
                primary = ct
                break

        # =========================
        # ✅ Anti-loop for train/car
        # =========================
        if primary:
            shared_missing = []
            if not shared.get("train"):
                shared_missing.append("train")
            if not shared.get("car"):
                shared_missing.append("car")

            if shared_missing:
                cnt = self._loop_bump(session, "ask_train_car")
                self._set_pending(session, scope="shared", slots=shared_missing)

                # 3-й раз подряд -> шаблон оперативникам
                if cnt >= 3:
                    session["pending"] = None
                    await self._save_session(chat_id_hash, session)
                    return BotReply(text=self._ops_template(primary))

                await self._save_session(chat_id_hash, session)
                return BotReply(text=self._train_car_question_for(primary))

        # collect missing per-case and submit
        for ct in ["lost", "complaint", "gratitude"]:
            for case in session["cases"]:
                if case["type"] != ct or case["status"] not in ("open", "collecting"):
                    continue

                # если готово -> submit
                if self._is_case_ready(session, case) and case["status"] != "open":
                    case_id = await self._submit_case(chat_id_hash, session, case)
                    await self._save_session(chat_id_hash, session)
                    return BotReply(text=f"Принял(а) ваше обращение: «{_case_title(ct)}». Номер заявки: {case_id}.")

                cs = case["slots"]

                if ct == "lost":
                    need = []
                    if not cs.get("place"):
                        need.append("place")
                    if not cs.get("item"):
                        need.append("item")
                    if not cs.get("when"):
                        need.append("when")
                    if need:
                        cnt = self._loop_bump(session, "ask_lost_bundle")
                        self._set_pending(session, scope="case", slots=need, case_type="lost")

                        if cnt >= 3:
                            session["pending"] = None
                            await self._save_session(chat_id_hash, session)
                            return BotReply(text=self._ops_template("lost"))

                        await self._save_session(chat_id_hash, session)
                        return BotReply(text=self._lost_bundle_question(angry=(is_angry or is_flood)))

                if ct == "complaint":
                    if not cs.get("complaintText"):
                        cnt = self._loop_bump(session, "ask_complaint_text")
                        self._set_pending(session, scope="case", slots=["complaintText"], case_type="complaint")

                        if cnt >= 3:
                            session["pending"] = None
                            await self._save_session(chat_id_hash, session)
                            return BotReply(text=self._ops_template("complaint"))

                        await self._save_session(chat_id_hash, session)
                        return BotReply(
                            text="Понял(а). Что именно случилось? (1–2 предложения, например: опоздал на 1 час / хамство / грязно / не работало отопление)."
                        )

                if ct == "gratitude":
                    if not cs.get("gratitudeText"):
                        cnt = self._loop_bump(session, "ask_gratitude_text")
                        self._set_pending(session, scope="case", slots=["gratitudeText"], case_type="gratitude")

                        if cnt >= 3:
                            session["pending"] = None
                            await self._save_session(chat_id_hash, session)
                            return BotReply(text=self._ops_template("gratitude"))

                        await self._save_session(chat_id_hash, session)
                        return BotReply(text="Понял(а). Напишите, пожалуйста, за что благодарите (1–2 предложения).")

        await self._save_session(chat_id_hash, session)
        return BotReply(text="Понял(а). Напишите детали одним сообщением, и я оформлю обращение.")
