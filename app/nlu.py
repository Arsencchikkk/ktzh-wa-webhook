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

    train = None
    car = None

    # Т58 / T58 / т 58
    m = re.search(r"\b[тt]\s*[-]?\s*(\d{1,4})\b", t)
    if m:
        train = f"Т{m.group(1)}"

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

    # flood: много одинаковых/частых сообщений
    last_ts = mod.get("last_ts", 0.0)
    dt = now - float(last_ts or 0.0)
    flooding = (dt < 2.0 and len(t) < 30) or repeat >= 2

    # aggression: грубые слова (минимально)
    angry_words = ("дебил", "идиот", "сука", "блять", "тупой", "прекрати")
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
        t = normalize(text)

        # cancel words
        cancel_words = ("стоп", "отмена", "прекрати", "хватит", "закрой", "не надо")
        cancel = any(w in t for w in cancel_words)

        # greeting
        greeting = bool(re.search(r"\b(привет|здравствуйте|здрасьте|добрый\s*(день|вечер|утро)|салам)\b", t))

        # intents
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

        # meaning score
        score = 0
        if intents:
            score += 2
        if train or car is not None:
            score += 1
        if len(t) >= 12:
            score += 1

        greeting_only = greeting and not intents and not slots and len(t.split()) <= 2

        return NluResult(
            intents=intents,
            slots=slots,
            greeting_only=greeting_only,
            cancel=cancel,
            meaning_score=score,
        )


def build_nlu() -> SimpleNLU:
    return SimpleNLU()
