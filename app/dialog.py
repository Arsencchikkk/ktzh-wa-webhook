from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import datetime as dt
import re
import secrets
import logging

from .nlu import build_nlu, extract_train_and_car, detect_aggression_and_flood, normalize

log = logging.getLogger("ktzh")


@dataclass
class BotReply:
    text: str
    meta: Dict[str, Any] | None = None


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _gen_case_id(prefix: str, chat_id_hash: str) -> str:
    d = _now_utc().strftime("%Y%m%d")
    short_chat = chat_id_hash[:6].upper()
    rnd = secrets.token_hex(3).upper()
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
    tn = normalize(text)

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


def _extract_train_car_any(text: str) -> Tuple[Optional[str], Optional[int]]:
    tr, car = extract_train_and_car(text)
    if not tr:
        tr = _extract_train_fallback(text)
    return tr, car


def _is_train_car_only(text: str) -> bool:
    tn = normalize(text)
    tr, car = _extract_train_car_any(text)  # ✅ оригинал
    tokens = re.findall(r"[a-zа-я0-9/]+", tn)
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
    tn = normalize(text)
    generic = ("хочу пожаловаться", "хочу жалобу", "хочу оставить жалобу", "у меня жалоба", "жалоба", "пожаловаться")
    return any(g in tn for g in generic) and len(tn.split()) <= 6


def _is_generic_gratitude(text: str) -> bool:
    tn = normalize(text)
    generic = ("хочу поблагодарить", "хочу сказать спасибо", "у меня благодарность", "благодарность", "спасибо")
    return any(g in tn for g in generic) and len(tn.split()) <= 6


def _is_delay_complaint(text: str) -> bool:
    tn = normalize(text)
    if any(k in tn for k in ("опозд", "опазд", "задерж")):
        return True
    return bool(re.search(r"\bна\s*\d+\s*(час|ч|минут|мин)\b", tn))


def _extract_place(text: str) -> Optional[str]:
    t = normalize(text)

    coupe = None
    seat = None

    m = re.search(r"\bкупе\s*(\d{1,2})\b", t)
    if m:
        coupe = m.group(1)

    m = re.search(r"\bместо\s*(\d{1,2})\b|\b(\d{1,2})\s*место\b", t)
    if m:
        seat = next((g for g in m.groups() if g and g.isdigit()), None)

    if "тамбур" in t:
        return "тамбур"

    if "верх" in t and "полк" in t:
        return "верхняя полка"
    if "ниж" in t and "полк" in t:
        return "нижняя полка"

    if coupe and seat:
        return f"купе {coupe}, место {seat}"
    if seat:
        return f"место {seat}"
    if coupe:
        return f"купе {coupe}"

    return None


def _extract_when(text: str) -> Optional[str]:
    tn = normalize(text)

    day = None
    if "сегодня" in tn:
        day = "сегодня"
    elif "вчера" in tn:
        day = "вчера"
    elif "позавчера" in tn:
        day = "позавчера"

    date = None
    m = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{2,4}))?\b", tn)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        date = f"{d.zfill(2)}.{mo.zfill(2)}.{y}" if y else f"{d.zfill(2)}.{mo.zfill(2)}"

    tm = None
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", tn)
    if m:
        tm = f"{m.group(1).zfill(2)}:{m.group(2)}"

    # склеиваем если есть
    if day and tm:
        return f"{day} {tm}"
    if date and tm:
        return f"{date} {tm}"
    return day or date or tm


def _extract_item(text: str) -> Optional[str]:
    tn = normalize(text)
    keys = (
        "сумк", "рюкзак", "чемодан", "пакет",
        "телефон", "документ", "паспорт",
        "кошелек", "бумажник", "наушник", "ноутбук",
        # ✅ одежда и частые вещи
        "кофт", "куртк", "одежд", "футболк", "штан", "джинс", "пальт", "шапк",
    )
    return _short(text) if any(k in tn for k in keys) else None


def _split_123(text: str) -> Dict[str, str]:
    """
    Парсим форматы:
      1) ...
      2) ...
      3) ...
    """
    s = (text or "").strip()
    out: Dict[str, str] = {}

    m1 = re.search(r"(?:^|\s)1\)\s*(.+?)(?=(?:\s*\b2\)\b|\s*$))", s, flags=re.S)
    if m1:
        out["1"] = m1.group(1).strip()

    m2 = re.search(r"(?:^|\s)2\)\s*(.+?)(?=(?:\s*\b3\)\b|\s*$))", s, flags=re.S)
    if m2:
        out["2"] = m2.group(1).strip()

    m3 = re.search(r"(?:^|\s)3\)\s*(.+?)\s*$", s, flags=re.S)
    if m3:
        out["3"] = m3.group(1).strip()

    return out


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
                "loop": {"key": None, "count": 0},
                "createdAt": _now_utc().isoformat(),
                "updatedAt": _now_utc().isoformat(),
            }
        if "loop" not in s:
            s["loop"] = {"key": None, "count": 0}
        return s

    async def _save_session(self, chat_id_hash: str, session: Dict[str, Any]) -> None:
        session["updatedAt"] = _now_utc().isoformat()
        if hasattr(self.store, "save_session"):
            await self.store.save_session(chat_id_hash, session)

    # ========== anti-loop ==========
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
            "2) Поезд № (например: Т78 или 10ЦА или 81/82 или ТЦ10):\n"
            "3) Маршрут (откуда–куда):\n"
            "4) Дата поездки (дд.мм.гггг):\n"
            "5) Время/примерно когда:\n"
            "6) Вагон № (если относится к вагону):\n"
            "7) Место/купе (если есть):\n"
            "8) Детали обращения (2–4 предложения):\n"
        )
        if case_type == "lost":
            return base.replace("8) Детали обращения", "8) Что потеряли + приметы (цвет/марка) и где оставили")
        return base

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
                "complaintTopic": None,   # "delay" | "service"
                "complaintWhen": None,    # дата/время (для delay)

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

    def _train_car_question_for(self, case_type: str, missing_train: bool, missing_car: bool) -> str:
        prefix = {
            "complaint": "Чтобы оформить жалобу",
            "gratitude": "Чтобы оформить благодарность",
            "lost": "Чтобы помочь найти вещь",
        }.get(case_type, "Чтобы продолжить")

        if missing_train and missing_car:
            return f"{prefix}, напишите номер поезда и вагон одним сообщением (пример: Т58, 7 вагон)."
        if missing_train:
            return f"{prefix}, напишите номер поезда (пример: Т58 или 10ЦА или 81/82 или ТЦ10)."
        if missing_car:
            return f"{prefix}, напишите номер вагона (пример: 7 вагон)."

        return f"{prefix}, уточните данные."

    def _lost_bundle_question(self, angry: bool = False) -> str:
        return (
            "Где в вагоне, что именно и когда оставили? Пример: место 12, черная сумка, вчера 14:30."
            if angry
            else "Для поиска вещи напишите одним сообщением: 1) где в вагоне (место/купе/полка/тамбур), 2) что за вещь и приметы, 3) когда примерно оставили."
        )

    def _is_case_ready(self, session: Dict[str, Any], case: Dict[str, Any]) -> bool:
        shared = session["shared"]
        cs = case["slots"]

        # train обязателен всегда
        if not shared.get("train"):
            return False

        # ✅ LOST: train+car + минимум 2 из 3 (место/вещь/когда)
        if case["type"] == "lost":
            if not shared.get("car"):
                return False
            filled = 0
            if cs.get("place"):
                filled += 1
            if cs.get("item"):
                filled += 1
            if cs.get("when"):
                filled += 1
            return filled >= 2

        # ✅ COMPLAINT: если delay — вагон не нужен, но нужен complaintWhen
        if case["type"] == "complaint":
            topic = cs.get("complaintTopic") or "service"
            if not cs.get("complaintText"):
                return False

            if topic == "delay":
                return bool(cs.get("complaintWhen"))
            # service / wagon-related
            return bool(shared.get("car"))

        # ✅ GRATITUDE: train+car+text
        if case["type"] == "gratitude":
            return bool(shared.get("car")) and bool(cs.get("gratitudeText"))

        return False

    async def _submit_case(self, chat_id_hash: str, session: Dict[str, Any], case: Dict[str, Any]) -> str:
        case_id = _gen_case_id("KTZH", chat_id_hash)
        case["caseId"] = case_id
        case["status"] = "open"
        case["openedAt"] = _now_utc().isoformat()

        if hasattr(self.store, "create_case"):
            await self.store.create_case({
                "caseId": case_id,
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
        changed = False

        if scope == "shared":
            if "train" in slots and train and not shared.get("train"):
                shared["train"] = train
                changed = True

            if "car" in slots:
                if car is not None and not shared.get("car"):
                    shared["car"] = car
                    changed = True
                else:
                    n = _is_only_number(text)
                    if n is not None and not shared.get("car"):
                        shared["car"] = n
                        changed = True

            ok = True
            if "train" in slots and not shared.get("train"):
                ok = False
            if "car" in slots and not shared.get("car"):
                ok = False
            if ok:
                session["pending"] = None
                if changed:
                    self._loop_reset(session)

        if scope == "case" and case_type:
            case = self._get_or_create_case(session, case_type)
            cs = case["slots"]

            parts = _split_123(text)

            # ✅ LOST: формат 1)/2)/3)
            if case_type == "lost" and parts:
                p1 = parts.get("1", "")
                p2 = parts.get("2", "")
                p3 = parts.get("3", "")

                if "place" in slots and not cs.get("place"):
                    pl = _extract_place(p1) or _extract_place(text)
                    if pl:
                        cs["place"] = pl
                        changed = True

                if "item" in slots and not cs.get("item"):
                    it = _short(p2) if p2 else (_extract_item(text) or None)
                    if it:
                        cs["item"] = it
                        changed = True

                if "when" in slots and not cs.get("when"):
                    wh = _extract_when(p3) or _extract_when(text)
                    if wh:
                        cs["when"] = wh
                        changed = True

            # ✅ complaintWhen (для delay)
            if case_type == "complaint":
                if "complaintWhen" in slots and not cs.get("complaintWhen"):
                    wh = _extract_when(text)
                    if wh:
                        cs["complaintWhen"] = wh
                        changed = True

            # fallback обычный
            if "place" in slots and not cs.get("place"):
                pl = _extract_place(text)
                if pl:
                    cs["place"] = pl
                    changed = True

            if "when" in slots and not cs.get("when"):
                wh = _extract_when(text)
                if wh:
                    cs["when"] = wh
                    changed = True

            if "item" in slots and not cs.get("item"):
                it = _extract_item(text)
                if it:
                    cs["item"] = it
                    changed = True

            if "complaintText" in slots and not cs.get("complaintText"):
                if (not _is_train_car_only(text)) and (not _is_generic_complaint(text)):
                    cs["complaintText"] = _short(text)
                    changed = True

            if "gratitudeText" in slots and not cs.get("gratitudeText"):
                if (not _is_train_car_only(text)) and (not _is_generic_gratitude(text)):
                    cs["gratitudeText"] = _short(text)
                    changed = True

            # ok check
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
                if changed:
                    self._loop_reset(session)

    async def handle(self, chat_id_hash: str, chat_meta: Dict[str, Any], user_text: str) -> BotReply:
        session = await self._load_session(chat_id_hash)

        session["chatId"] = str(chat_meta.get("chatId") or session.get("chatId") or "")
        session["channelId"] = str(chat_meta.get("channelId") or session.get("channelId") or "")
        session["chatType"] = str(chat_meta.get("chatType") or session.get("chatType") or "")

        text = user_text or ""
        tnorm = normalize(text)

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

        if session.get("pending"):
            self._apply_pending(session, text)

        if getattr(nlu_res, "greeting_only", False) and not session.get("cases"):
            await self._save_session(chat_id_hash, session)
            return BotReply(text="Здравствуйте! Опишите проблему одним сообщением (опоздание / забытая вещь / жалоба / благодарность).")

        intents: List[str] = list(getattr(nlu_res, "intents", []) or [])

        shared = session["shared"]
        slots = getattr(nlu_res, "slots", {}) or {}

        # apply NLU slots
        if slots.get("train") and not shared.get("train"):
            shared["train"] = slots["train"]
            self._loop_reset(session)
        if slots.get("car") and not shared.get("car"):
            shared["car"] = slots["car"]
            self._loop_reset(session)

        # apply regex slots
        tr, car = _extract_train_car_any(text)
        if tr and not shared.get("train"):
            shared["train"] = tr
            self._loop_reset(session)
        if car is not None and not shared.get("car"):
            shared["car"] = car
            self._loop_reset(session)

        # create cases
        for it in intents:
            self._get_or_create_case(session, it)

        # ===== complaint fill + topic + when =====
        if "complaint" in intents:
            ccase = self._get_or_create_case(session, "complaint")

            if not ccase["slots"].get("complaintText"):
                if (not _is_train_car_only(text)) and (not _is_generic_complaint(text)):
                    ccase["slots"]["complaintText"] = _short(text)

            # topic — по самому complaintText (если есть), иначе по текущему сообщению
            base_text = ccase["slots"].get("complaintText") or text
            if not ccase["slots"].get("complaintTopic"):
                ccase["slots"]["complaintTopic"] = "delay" if _is_delay_complaint(base_text) else "service"

            # для delay — пробуем сразу вытащить когда
            if ccase["slots"].get("complaintTopic") == "delay" and not ccase["slots"].get("complaintWhen"):
                wh = _extract_when(text)
                if wh:
                    ccase["slots"]["complaintWhen"] = wh

        # ===== gratitude fill =====
        if "gratitude" in intents:
            gcase = self._get_or_create_case(session, "gratitude")
            if not gcase["slots"].get("gratitudeText"):
                if (not _is_train_car_only(text)) and (not _is_generic_gratitude(text)):
                    gcase["slots"]["gratitudeText"] = _short(text)
            if slots.get("staffName") and not gcase["slots"].get("staffName"):
                gcase["slots"]["staffName"] = str(slots["staffName"])

        # ===== lost fill =====
        if "lost" in intents:
            lcase = self._get_or_create_case(session, "lost")

            parts = _split_123(text)
            if parts:
                p1 = parts.get("1", "")
                p2 = parts.get("2", "")
                p3 = parts.get("3", "")

                if not lcase["slots"].get("place"):
                    lcase["slots"]["place"] = _extract_place(p1) or _extract_place(text)
                if not lcase["slots"].get("item"):
                    lcase["slots"]["item"] = _short(p2) if p2 else (_extract_item(text) or None)
                if not lcase["slots"].get("when"):
                    lcase["slots"]["when"] = _extract_when(p3) or _extract_when(text)
            else:
                if not lcase["slots"].get("place"):
                    lcase["slots"]["place"] = _extract_place(text)
                if not lcase["slots"].get("item"):
                    lcase["slots"]["item"] = _extract_item(text)
                if not lcase["slots"].get("when"):
                    lcase["slots"]["when"] = _extract_when(text)

        # primary
        primary: Optional[str] = None
        for ct in ["lost", "complaint", "gratitude"]:
            if any(c["type"] == ct and c["status"] in ("open", "collecting") for c in session["cases"]):
                primary = ct
                break

        # ===== ask train/car (но для delay не спрашиваем car) =====
        if primary:
            missing_train = not bool(shared.get("train"))
            missing_car = not bool(shared.get("car"))

            if primary == "complaint":
                ccase = self._get_or_create_case(session, "complaint")
                if ccase["slots"].get("complaintTopic") == "delay":
                    missing_car = False  # ✅ delay -> вагон не нужен

            if missing_train or missing_car:
                cnt = self._loop_bump(session, "ask_train_car")
                self._set_pending(
                    session,
                    scope="shared",
                    slots=[s for s in ["train", "car"] if (s == "train" and missing_train) or (s == "car" and missing_car)],
                )

                if cnt >= 3:
                    session["pending"] = None
                    await self._save_session(chat_id_hash, session)
                    return BotReply(text=self._ops_template(primary))

                await self._save_session(chat_id_hash, session)
                return BotReply(text=self._train_car_question_for(primary, missing_train, missing_car))

            self._loop_reset(session)

        # ===== collect missing + submit =====
        for ct in ["lost", "complaint", "gratitude"]:
            for case in session["cases"]:
                if case["type"] != ct or case["status"] not in ("open", "collecting"):
                    continue

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

                    # ✅ важно: если уже есть 2 из 3 — мы не спрашиваем третье, т.к. _is_case_ready уже бы сработал
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
                    topic = cs.get("complaintTopic") or "service"

                    if topic == "delay" and not cs.get("complaintWhen"):
                        cnt = self._loop_bump(session, "ask_complaint_when")
                        self._set_pending(session, scope="case", slots=["complaintWhen"], case_type="complaint")

                        if cnt >= 3:
                            session["pending"] = None
                            await self._save_session(chat_id_hash, session)
                            return BotReply(text=self._ops_template("complaint"))

                        await self._save_session(chat_id_hash, session)
                        return BotReply(text="Уточните, пожалуйста, дату поездки и примерное время (например: вчера 19:00 или 01.02.2026 18:30).")

                    if not cs.get("complaintText"):
                        cnt = self._loop_bump(session, "ask_complaint_text")
                        self._set_pending(session, scope="case", slots=["complaintText"], case_type="complaint")

                        if cnt >= 3:
                            session["pending"] = None
                            await self._save_session(chat_id_hash, session)
                            return BotReply(text=self._ops_template("complaint"))

                        await self._save_session(chat_id_hash, session)
                        return BotReply(text="Понял(а). Что именно случилось? (1–2 предложения, например: опоздал на 1 час / хамство / грязно / не работало отопление).")

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
