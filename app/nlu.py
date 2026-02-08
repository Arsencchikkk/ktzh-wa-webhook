from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import time


def normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("ё", "е")
    return t


def extract_train_and_car(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Ловим поезд в форматах:
      - Т58 / т 58 / T58
      - 10ЦА / 123А / 81/82
      - тц10 / TC10 / ца80  (буквы+цифры)  ✅ добавлено
    """
    t = normalize(text)

    train: Optional[str] = None
    car: Optional[int] = None

    # ===== TRAIN =====

    # 1) Т58 / T58 / т 58 / т-58
    m = re.search(r"\b[тt]\s*[-]?\s*(\d{1,4})\b", t)
    if m:
        train = f"Т{m.group(1)}".upper()

    # 2) 81/82, 10ца, 123а (цифры + буквы)
    # ВАЖНО: не спутать с "8 вагон"
    if not train:
        # если рядом явно "вагон" — не считаем это поездом
        if not re.search(r"\bвагон\s*\d{1,2}\b|\b\d{1,2}\s*вагон\b", t):
            m = re.search(r"\b(\d{2,4}(?:\s*/\s*\d{2,4})?[a-zа-я]{0,3})\b", t)
            if m:
                cand = m.group(1).replace(" ", "").upper()
                if not re.fullmatch(r"\d{1,2}", cand):
                    train = cand

    # 3) тц10 / TC10 / ца80 (буквы + цифры) ✅
    if not train:
        m = re.search(r"\b([a-zа-я]{1,3})\s*[-]?\s*(\d{1,4})\b", t)
        if m:
            letters = m.group(1)
            digits = m.group(2)
            # чтобы не ловить мусорные слова
            if letters not in {"ваг", "вагон", "мест", "место", "куп", "купе"}:
                train = f"{letters}{digits}".upper()

    # ===== CAR (вагон) =====
    m = re.search(r"\bвагон\s*(\d{1,2})\b|\b(\d{1,2})\s*вагон\b", t)
    if m:
        num = next((g for g in m.groups() if g), None)
        if num:
            car = int(num)

    return train, car


def detect_aggression_and_flood(session: Dict[str, Any], text: str) -> Tuple[Dict[str, Any], bool, bool]:
    t = normalize(text)

    mod = session.get("moderation") or {}
    now = time.time()

    prev_text = mod.get("prev_text")
    repeat = mod.get("repeat_count", 0)

    if prev_text and prev_text == t:
        repeat += 1
    else:
        repeat = 0

    last_ts = mod.get("last_ts", 0.0)
    dt_sec = now - float(last_ts or 0.0)
    flooding = (dt_sec < 2.0 and len(t) < 30) or repeat >= 2

    # ✅ "прекрати" НЕ считаем агрессией (это cancel)
    angry_words = ("дебил", "идиот", "сука", "блять", "тупой")
    angry = any(w in t for w in angry_words)

    mod["prev_text"] = t
    mod["repeat_count"] = repeat
    mod["last_ts"] = now
    session["moderation"] = mod

    return session, angry, flooding


@dataclass
class NluResult:
    intents: List[str]
    slots: Dict[str, Any]
    greeting_only: bool
    cancel: bool
    meaning_score: int


class SimpleNLU:
    GRATITUDE_KEYS = ("благодар", "поблагодар", "спасибо", "рахмет")
    LOST_KEYS = ("потерял", "потеряла", "забыл", "забыла", "оставил", "оставила", "пропал", "вещ", "сумк", "рюкзак", "чемодан")

    COMPLAINT_KEYS = (
        "жалоб", "пожалов", "претенз", "недовол",
        "опозд", "опоздан", "опоздал", "опоздала",
        "опазд", "опаздал", "опаздала", "опазды",
        "задерж", "задержк",
        "грязн", "хам", "не работает", "не работал", "сломал", "плохо", "ужас", "беспредел",
    )

    COMPLAINT_HINTS = ("опаз", "задерж", "час", "мин", "беспредел", "ужас", "кошмар", "невыносимо")

    def analyze(self, text: str) -> NluResult:
        orig = (text or "").strip()
        t = normalize(orig)

        cancel_words = ("стоп", "отмена", "прекрати", "хватит", "закрой", "не надо")
        cancel = any(w in t for w in cancel_words)

        greeting_only = bool(
            re.fullmatch(r"(привет|здравствуйте|здрасьте|салам|добрый\s*(день|вечер|утро))", t)
        )

        intents: List[str] = []

        if any(k in t for k in self.GRATITUDE_KEYS):
            intents.append("gratitude")

        if any(k in t for k in self.LOST_KEYS):
            intents.append("lost")

        if any(k in t for k in self.COMPLAINT_KEYS) or any(h in t for h in self.COMPLAINT_HINTS):
            intents.append("complaint")

        train, car = extract_train_and_car(orig)
        slots: Dict[str, Any] = {}
        if train:
            slots["train"] = train
        if car is not None:
            slots["car"] = car

        # staffName (проводника Аймуратова / кассира Иванова)
        m = re.search(
            r"(проводник\w*|кассир\w*|сотрудник\w*|начальник\w*\s*поезда?)\s+([А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+){0,2})",
            orig,
        )
        if m:
            slots["staffName"] = m.group(2).strip()

        score = 0
        if intents:
            score += 2
        if train or car is not None:
            score += 1
        if len(t) >= 12:
            score += 1

        return NluResult(
            intents=intents,
            slots=slots,
            greeting_only=greeting_only,
            cancel=cancel,
            meaning_score=score,
        )


def build_nlu() -> SimpleNLU:
    return SimpleNLU()
