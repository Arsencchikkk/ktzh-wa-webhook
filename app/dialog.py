from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import datetime as dt
import hashlib
import re

from .nlu import build_nlu, extract_train_and_car, detect_aggression_and_flood, normalize


@dataclass
class BotReply:
    text: str
    meta: Dict[str, Any] | None = None


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _gen_case_id(prefix: str, chat_id_hash: str) -> str:
    # KTZH-20260207-66AC3D18 style
    d = _now_utc().strftime("%Y%m%d")
    h = hashlib.sha256((prefix + "|" + d + "|" + chat_id_hash).encode("utf-8")).hexdigest().upper()
    return f"{prefix}-{d}-{h[:8]}"


def _short(text: str) -> str:
    return (text or "").strip()


def _is_only_number(text: str) -> Optional[int]:
    t = normalize(text)
    m = re.fullmatch(r"\s*(\d{1,2})\s*", t)
    if not m:
        return None
    v = int(m.group(1))
    if 1 <= v <= 99:
        return v
    return None


def _extract_place(text: str) -> Optional[str]:
    t = normalize(text)
    # место 12 / 12 место
    m = re.search(r"\bместо\s*(\d{1,2})\b|\b(\d{1,2})\s*место\b", t)
    if m:
        num = next((g for g in m.groups() if g and g.isdigit()), None)
        return f"место {num}" if num else None

    # купе 3
    m = re.search(r"\bкупе\s*(\d{1,2})\b", t)
    if m:
        return f"купе {m.group(1)}"

    # тамбур
    if "тамбур" in t:
        return "тамбур"

    # верхняя/нижняя полка
    if "верх" in t and "полк" in t:
        return "верхняя полка"
    if "ниж" in t and "полк" in t:
        return "нижняя полка"

    return None


def _extract_when(text: str) -> Optional[str]:
    t = normalize(text)
    # простые маркеры
    if "сегодня" in t:
        return "сегодня"
    if "вчера" in t:
        return "вчера"
    if "позавчера" in t:
        return "позавчера"

    # дата 07.02 / 7.02 / 07-02
    m = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{2,4}))?\b", t)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if y:
            return f"{d.zfill(2)}.{mo.zfill(2)}.{y}"
        return f"{d.zfill(2)}.{mo.zfill(2)}"

    # время 14:30
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        return f"{m.group(1).zfill(2)}:{m.group(2)}"

    return None


def _extract_item(text: str) -> Optional[str]:
    t = normalize(text)
    # если есть "сумк/рюкзак/чемодан/пакет/телефон/документы"
    keys = ["сумк", "рюкзак", "чемодан", "пакет", "телефон", "документ", "паспорт", "кошелек", "бумажник", "наушник", "ноутбук"]
    if any(k in t for k in keys):
        return _short(text)
    return None


def _required_slots(case_type: str) -> List[str]:
    if case_type == "lost":
        return ["train", "car", "place", "item", "when"]
    if case_type == "complaint":
        return ["train", "car", "complaintText"]
    if case_type == "gratitude":
        return ["train", "car", "gratitudeText"]  # staffName optional
    return []


def _case_title(case_type: str) -> str:
    return {
        "lost": "Забытые/потерянные вещи",
        "complaint": "Жалоба",
        "gratitude": "Благодарность",
    }.get(case_type, case_type)


class DialogManager:
    """
    Универсальный rule-based менеджер диалога:
    - shared slots: train/car
    - multi-intent -> несколько кейсов
    - pending-slot -> короткие ответы трактуем правильно
    """

    def __init__(self, store: Any):
        self.store = store
        self.nlu = build_nlu()

    # ---- storage helpers ----

    def _load_session(self, chat_id_hash: str) -> Dict[str, Any]:
        s = None
        if hasattr(self.store, "get_session"):
            s = self.store.get_session(chat_id_hash)
        if not s:
            s = {
                "shared": {"train": None, "car": None},
                "cases": [],          # list[dict]
                "pending": None,      # dict with expected slots
                "aggression": 0,
                "flood": None,
                "createdAt": _now_utc().isoformat(),
                "updatedAt": _now_utc().isoformat(),
            }
        return s

    def _save_session(self, chat_id_hash: str, session: Dict[str, Any]) -> None:
        session["updatedAt"] = _now_utc().isoformat()
        if hasattr(self.store, "save_session"):
            self.store.save_session(chat_id_hash, session)

    # ---- case helpers ----

    def _get_or_create_case(self, session: Dict[str, Any], case_type: str) -> Dict[str, Any]:
        for c in session["cases"]:
            if c["type"] == case_type and c["status"] in ("open", "collecting"):
                return c
        c = {
            "type": case_type,
            "status": "collecting",
            "slots": {"place": None, "item": None, "when": None, "complaintText": None, "gratitudeText": None, "staffName": None},
            "caseId": None,
            "createdAt": _now_utc().isoformat(),
        }
        session["cases"].append(c)
        return c

    def _close_all_cases(self, session: Dict[str, Any], reason: str) -> None:
        for c in session["cases"]:
            if c["status"] in ("open", "collecting"):
                c["status"] = "closed"
                c["closeReason"] = reason
                c["closedAt"] = _now_utc().isoformat()
        session["pending"] = None

    # ---- pending slot ----

    def _set_pending(self, session: Dict[str, Any], scope: str, slots: List[str], case_type: Optional[str] = None) -> None:
        session["pending"] = {"scope": scope, "slots": slots, "caseType": case_type}

    def _apply_pending(self, session: Dict[str, Any], text: str) -> bool:
        p = session.get("pending")
        if not p:
            return False

        scope = p.get("scope")
        slots: List[str] = p.get("slots") or []
        case_type = p.get("caseType")

        # сначала попробуем вытащить поезд/вагон
        train, car = extract_train_and_car(text)
        shared = session["shared"]

        changed = False

        # если ждём shared train/car
        if scope == "shared":
            if "train" in slots and train and not shared.get("train"):
                shared["train"] = train
                changed = True
            if "car" in slots:
                if car is not None and not shared.get("car"):
                    shared["car"] = car
                    changed = True
                else:
                    # короткий ответ "8" -> вагон
                    n = _is_only_number(text)
                    if n is not None and not shared.get("car"):
                        shared["car"] = n
                        changed = True

        # если ждём case-specific
        if scope == "case" and case_type:
            case = self._get_or_create_case(session, case_type)
            cs = case["slots"]

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
                # если человек написал только "Т58 7 вагон" — это НЕ complaintText
                t = normalize(text)
                tr2, car2 = extract_train_and_car(t)
                if not (tr2 or car2) or len(t.split()) > 3:
                    cs["complaintText"] = _short(text)
                    changed = True

            if "gratitudeText" in slots and not cs.get("gratitudeText"):
                cs["gratitudeText"] = _short(text)
                changed = True

            if "staffName" in slots and not cs.get("staffName"):
                cs["staffName"] = _short(text)
                changed = True

        # pending закрываем только если все слоты заполнены
        if scope == "shared":
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

        return changed

    # ---- question builder ----

    def _bundle_train_car_question(self, angry: bool = False) -> str:
        if angry:
            return "Напишите номер поезда и вагон одним сообщением. Пример: Т58, 7 вагон."
        return "Уточните номер поезда и номер вагона одним сообщением (пример: Т58, 7 вагон)."

    def _lost_bundle_question(self, angry: bool = False) -> str:
        if angry:
            return "Где в вагоне, что именно и когда оставили? Пример: место 12, черная сумка, вчера 14:30."
        return "Для поиска вещи напишите одним сообщением: 1) где в вагоне (место/купе/полка/тамбур), 2) что за вещь и приметы, 3) когда примерно оставили."

    def _gratitude_bundle_question(self, angry: bool = False) -> str:
        if angry:
            return "Кого благодарите и за что? + поезд и вагон (если знаете)."
        return "Кого хотите поблагодарить (если знаете — имя/должность) и за что? Если знаете — добавьте поезд и вагон."

    # ---- submit when ready ----

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

    def _submit_case(self, chat_id_hash: str, session: Dict[str, Any], case: Dict[str, Any]) -> str:
        case_id = _gen_case_id("KTZH", chat_id_hash)
        case["caseId"] = case_id
        case["status"] = "open"
        case["openedAt"] = _now_utc().isoformat()

        # здесь можно дернуть вашу CRM/внутренний API,
        # а пока просто сохраняем в session/store
        return case_id

    # ---- main ----

    def handle(self, chat_id_hash: str, chat_meta: Dict[str, Any], user_text: str) -> BotReply:
        session = self._load_session(chat_id_hash)

        # агрессия/флуд
        _, is_angry, is_flood = detect_aggression_and_flood(session, user_text)

        text = user_text or ""
        tnorm = normalize(text)

        # авто-отмена / нашлась вещь
        nlu_res = self.nlu.analyze(text)
        if nlu_res.cancel:
            self._close_all_cases(session, reason="user_cancel_or_found")
            self._save_session(chat_id_hash, session)
            msg = "Понял(а). Закрыл(а) обращение(я). Если потребуется — напишите снова."
            return BotReply(text=msg)

        # 1) pending-slot: сначала пытаемся “пришить” короткий ответ
        if session.get("pending"):
            self._apply_pending(session, text)

        # 2) greeting-only без смысла => не создаём кейс
        if nlu_res.greeting_only and nlu_res.meaning_score < 2 and not session.get("cases"):
            self._save_session(chat_id_hash, session)
            return BotReply(text="Здравствуйте! Опишите проблему одним сообщением (например: поезд опоздал / забыл вещь / жалоба / благодарность).")

        # 3) обновляем shared train/car, если нашли
        shared = session["shared"]
        if nlu_res.slots.get("train") and not shared.get("train"):
            shared["train"] = nlu_res.slots["train"]
        if nlu_res.slots.get("car") and not shared.get("car"):
            shared["car"] = nlu_res.slots["car"]

        # 4) если смысла мало и нет pending — просим уточнить
        if nlu_res.meaning_score < 2 and not session.get("pending"):
            self._save_session(chat_id_hash, session)
            if is_angry or is_flood:
                return BotReply(text="Напишите конкретно, что случилось. Пример: 'Т58, 7 вагон, поезд опоздал на 40 минут' или 'оставил сумку, место 12'.")
            return BotReply(text="Я вас понял(а). Уточните, пожалуйста, что именно случилось (опоздание / забытая вещь / жалоба / благодарность).")

        # 5) создаём кейсы по интентам (multi-intent)
        intents = nlu_res.intents
        if not intents:
            # если нет интента, но есть поезд/вагон, будем добирать “что случилось”
            intents = []

        for it in intents:
            self._get_or_create_case(session, it)

        # 6) если человек написал “спасибо/благодарность”, НЕ превращаем в жалобу.
        # Но и НЕ закрываем: добираем детали.
        # Заполним текст благодарности, если похоже
        if "gratitude" in intents:
            gcase = self._get_or_create_case(session, "gratitude")
            if not gcase["slots"].get("gratitudeText"):
                gcase["slots"]["gratitudeText"] = _short(text)

        if "complaint" in intents:
            ccase = self._get_or_create_case(session, "complaint")
            # complaintText — только если это реально описание, а не только поезд/вагон
            if not ccase["slots"].get("complaintText"):
                tr, car = extract_train_and_car(text)
                pure_train_car = bool(tr or car) and len(_tokens := re.findall(r"[a-zа-я0-9]+", tnorm)) <= 4
                if not pure_train_car:
                    ccase["slots"]["complaintText"] = _short(text)

        if "lost" in intents:
            lcase = self._get_or_create_case(session, "lost")
            if not lcase["slots"].get("place"):
                lcase["slots"]["place"] = _extract_place(text)
            if not lcase["slots"].get("item"):
                lcase["slots"]["item"] = _extract_item(text)
            if not lcase["slots"].get("when"):
                lcase["slots"]["when"] = _extract_when(text)

        # 7) Определяем следующий вопрос:
        #    - сначала shared train+car (один раз для всех кейсов)
        #    - потом кейс-специфичные бандлы (lost/place+item+when; gratitude/who+for what)
        shared_missing = []
        if not shared.get("train"):
            shared_missing.append("train")
        if not shared.get("car"):
            shared_missing.append("car")

        if shared_missing:
            # ставим pending на shared
            self._set_pending(session, scope="shared", slots=shared_missing)
            self._save_session(chat_id_hash, session)
            return BotReply(text=self._bundle_train_car_question(angry=(is_angry or is_flood)))

        # shared есть — добираем по кейсам
        # приоритет: lost -> complaint -> gratitude (потому что lost срочнее)
        ordered = ["lost", "complaint", "gratitude"]
        for ct in ordered:
            for case in session["cases"]:
                if case["type"] != ct or case["status"] not in ("open", "collecting"):
                    continue

                # если готов — открываем/создаём заявку
                if self._is_case_ready(session, case) and case["status"] != "open":
                    case_id = self._submit_case(chat_id_hash, session, case)
                    self._save_session(chat_id_hash, session)
                    return BotReply(text=f"Принял(а) ваше обращение: «{_case_title(ct)}». Номер заявки: {case_id}.")

                # если не готов — задаём один “правильный” вопрос-бандл
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
                        self._set_pending(session, scope="case", slots=need, case_type="lost")
                        self._save_session(chat_id_hash, session)
                        return BotReply(text=self._lost_bundle_question(angry=(is_angry or is_flood)))

                if ct == "complaint":
                    if not cs.get("complaintText"):
                        self._set_pending(session, scope="case", slots=["complaintText"], case_type="complaint")
                        self._save_session(chat_id_hash, session)
                        if is_angry or is_flood:
                            return BotReply(text="Коротко: что именно случилось? (1–2 предложения).")
                        return BotReply(text="Опишите, пожалуйста, суть жалобы (что именно случилось).")

                if ct == "gratitude":
                    # gratitudeText уже может быть, но нам нужны детали: кого благодарит/за что
                    # train/car уже есть
                    if not cs.get("gratitudeText") or len(cs.get("gratitudeText", "")) < 4:
                        self._set_pending(session, scope="case", slots=["gratitudeText"], case_type="gratitude")
                        self._save_session(chat_id_hash, session)
                        return BotReply(text=self._gratitude_bundle_question(angry=(is_angry or is_flood)))
                    # попросим "кого благодарит" как staffName (не обязателен, но улучшает)
                    if not cs.get("staffName"):
                        self._set_pending(session, scope="case", slots=["staffName"], case_type="gratitude")
                        self._save_session(chat_id_hash, session)
                        if is_angry or is_flood:
                            return BotReply(text="Кого благодарите? (имя/должность, если знаете)")
                        return BotReply(text="Кого именно хотите поблагодарить? (если знаете — имя/должность)")

                    # теперь можно открыть заявку благодарности
                    if case["status"] != "open":
                        case_id = self._submit_case(chat_id_hash, session, case)
                        self._save_session(chat_id_hash, session)
                        return BotReply(text=f"Принял(а) вашу благодарность. Номер заявки: {case_id}.")

        # если дошли сюда — просто подтверждаем
        self._save_session(chat_id_hash, session)
        if is_angry or is_flood:
            return BotReply(text="Понял. Напишите детали одним сообщением, чтобы я оформил обращение.")
        return BotReply(text="Понял(а). Напишите детали одним сообщением, и я оформлю обращение.")
