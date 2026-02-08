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
    t = normalize(text)

    train: Optional[str] = None
    car: Optional[int] = None

    # 1) Т58 / T58 / т 58
    m = re.search(r"\b[тt]\s*[-]?\s*(\d{1,4})\b", t)
    if m:
        train = f"Т{m.group(1)}"

    # 2) поезд / № / цифры+буквы: 10ЦА, 81/82, 123А
    if not train:
        m = re.search(
            r"(?:\bпоезд\b|\b№\b|\bn\b)?\s*"
            r"\b(\d{2,4}(?:/\d{2,4})?[a-zа-я]{0,3})\b",
            t,
        )
        if m:
            cand = m.group(1).upper()
            # защита: не спутать с "8 вагон"
            if not re.fullmatch(r"\d{1,2}", cand):
                train = cand

    # вагон 9 / 9 вагон
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
    def analyze(self, text: str) -> NluResult:
        orig = (text or "").strip()
        t = normalize(orig)

        cancel_words = ("стоп", "отмена", "прекрати", "хватит", "закрой", "не надо")
        cancel = any(w in t for w in cancel_words)

        greeting = bool(re.search(r"\b(привет|здравствуйте|здрасьте|добрый\s*(день|вечер|утро)|салам)\b", t))
        # ✅ greeting_only — только если реально одно приветствие
        greeting_only = bool(re.fullmatch(r"(привет|здравствуйте|здрасьте|салам|добрый\s*(день|вечер|утро))", t))

        intents: List[str] = []
        if any(k in t for k in ("благодар", "поблагодар", "спасибо", "рахмет")):
            intents.append("gratitude")

        if any(k in t for k in ("потерял", "забыл", "оставил", "пропал", "вещ", "сумк", "рюкзак", "чемодан")):
            intents.append("lost")

        if any(k in t for k in ("жалоб", "опозд", "задерж", "ужас", "грязн", "хам", "не работает", "сломал", "плохо")):
            intents.append("complaint")

        train, car = extract_train_and_car(t)
        slots: Dict[str, Any] = {}
        if train:
            slots["train"] = train
        if car is not None:
            slots["car"] = car

        # ✅ staffName (проводника Аймуратова / кассира Иванова)
        m = re.search(
            r"(проводник\w*|кассир\w*|сотрудник\w*|начальник\w*\s*поезда?)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})",
            orig
        )
        if m:
            slots["staffName"] = m.group(2)

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
