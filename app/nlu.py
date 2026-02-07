from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re


# ---------- patterns ----------
TRAIN_RE = re.compile(r"(?i)\b[тt]\s*[-]?\s*(\d{1,4})\b")
CAR_RE_1 = re.compile(r"(?i)\bвагон\s*№?\s*(\d{1,2})\b")
CAR_RE_2 = re.compile(r"(?i)\b(\d{1,2})\s*вагон\b")
JUST_INT_RE = re.compile(r"^\s*(\d{1,2})\s*$")

GREETINGS = {
    "привет", "здравствуйте", "здрасте", "салам", "сәлем", "ассаламуалейкум",
    "добрый день", "доброе утро", "добрый вечер", "hi", "hello"
}

RESET_WORDS = {"сброс", "reset", "отмена", "стоп", "начать заново"}

GRATITUDE_WORDS = {
    "благодарность", "спасибо", "рахмет", "алғыс", "хочу поблагодарить", "благодарю",
    "выражаю благодарность"
}

LOST_WORDS = {
    "потерял", "потеряла", "забыл", "забыла", "оставил", "оставила", "пропала", "пропал",
    "сумка", "рюкзак", "кошелек", "кошелёк", "телефон", "документы", "паспорт", "карта", "ноутбук"
}

COMPLAINT_WORDS = {
    "жалоба", "плохо", "ужас", "не работает", "грязно", "хамство", "грубость",
    "опоздал", "опоздание", "задержка", "задержали", "сломано", "холодно", "жарко",
    "не дали", "не пустили", "проводник", "контролер", "контролёр"
}

DELAY_WORDS = {"опоздал", "опоздание", "задержка", "задержали", "задержался", "прибыл поздно"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_train(text: str) -> Optional[str]:
    m = TRAIN_RE.search(text)
    if not m:
        return None
    return f"T{m.group(1)}"


def extract_car(text: str) -> Optional[int]:
    for rx in (CAR_RE_1, CAR_RE_2):
        m = rx.search(text)
        if m:
            try:
                v = int(m.group(1))
                if 1 <= v <= 99:
                    return v
            except ValueError:
                pass
    # If message is just "7"
    m = JUST_INT_RE.match(text.strip())
    if m:
        v = int(m.group(1))
        if 1 <= v <= 99:
            return v
    return None


def extract_train_and_car(text: str) -> Tuple[Optional[str], Optional[int]]:
    return extract_train(text), extract_car(text)


def is_greeting_only(text: str) -> bool:
    t = normalize(text)
    if not t:
        return True
    # if contains only greeting words and punctuation
    t2 = re.sub(r"[^a-zа-яёқңүұһәі ]+", " ", t)
    t2 = normalize(t2)
    return any(t2 == g or t2.startswith(g + " ") or t2.endswith(" " + g) for g in GREETINGS)


def contains_any(text: str, words: set[str]) -> bool:
    t = normalize(text)
    return any(w in t for w in words)


@dataclass
class NLUResult:
    intents: List[str]            # possibly multi
    train: Optional[str] = None
    carNumber: Optional[int] = None
    # free text signals
    complaintText: Optional[str] = None
    gratitudeText: Optional[str] = None
    lostHint: Optional[str] = None
    meaningScore_0_100: int = 0
    isGreetingOnly: bool = False
    isReset: bool = False


def meaning_score(text: str, pending_slots: List[str]) -> int:
    """
    Score "has meaning" but: if we are waiting for something, short answers can be meaningful.
    """
    raw = (text or "").strip()
    if not raw:
        return 0

    t = normalize(raw)

    # If we have pending slots, even "8" can be meaningful
    if pending_slots:
        # train/car shortcuts
        if extract_train(raw) or extract_car(raw):
            return 90
        # any short non-empty response -> medium
        if len(t) <= 12:
            return 70

    # greeting-only
    if is_greeting_only(raw):
        return 0

    # explicit keywords
    if contains_any(raw, GRATITUDE_WORDS | LOST_WORDS | COMPLAINT_WORDS):
        return 90

    # train/car mention
    if extract_train(raw) or extract_car(raw):
        return 75

    # very short messages: "ок", "да", "нет"
    if len(t) <= 2:
        return 5
    if len(t) <= 5 and t in {"ок", "окей", "да", "нет", "ага"}:
        return 10

    # longer text => more meaning
    if len(t) >= 15:
        return 60
    return 35


def classify_intents(text: str) -> List[str]:
    """
    Hard overrides:
      - if gratitude present -> gratitude intent exists
      - if lost present -> lost intent exists
      - if complaint present -> complaint intent exists
    Allow multi-intent.
    """
    t = normalize(text)
    intents: List[str] = []

    if contains_any(t, GRATITUDE_WORDS):
        intents.append("gratitude")

    # lost: if lost words OR explicit phrases "оставил в поезде", "потерял"
    if contains_any(t, LOST_WORDS) or "оставил в поезде" in t or "потерял" in t or "забыл" in t:
        intents.append("lost_and_found")

    if contains_any(t, COMPLAINT_WORDS) or contains_any(t, DELAY_WORDS):
        intents.append("complaint")

    # de-dup preserve order
    seen = set()
    out = []
    for x in intents:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_nlu(text: str, pending_slots: List[str]) -> NLUResult:
    t = (text or "").strip()
    is_reset = contains_any(t, RESET_WORDS)
    greet = is_greeting_only(t)
    ms = meaning_score(t, pending_slots)

    train, car = extract_train_and_car(t)
    intents = [] if greet else classify_intents(t)

    res = NLUResult(
        intents=intents,
        train=train,
        carNumber=car,
        meaningScore_0_100=ms,
        isGreetingOnly=greet,
        isReset=is_reset,
    )

    # Attach free text (don’t overthink: rules only)
    if "complaint" in intents:
        res.complaintText = t
    if "gratitude" in intents:
        res.gratitudeText = t
    if "lost_and_found" in intents:
        res.lostHint = t

    return res
