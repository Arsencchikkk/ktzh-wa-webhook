from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import time

from .settings import settings


# -------------------------
# Normalization / tokens
# -------------------------

def normalize(text: str) -> str:
    t = (text or "").strip().lower()
    return t.replace("ё", "е")


def _tokens(text: str) -> List[str]:
    t = normalize(text)
    return re.findall(r"[a-zа-я0-9]+", t)


# -------------------------
# Simple regex extractors
# -------------------------

RE_TRAIN = re.compile(r"(?i)\b[тt]\s*[-]?\s*(\d{1,4})\b")
RE_CAR = re.compile(r"(?i)\b(\d{1,2})\s*(вагон|вгн|ваг)\b|\bвагон\s*(\d{1,2})\b")
RE_NUM = re.compile(r"\b(\d{1,4})\b")


def extract_train_and_car(text: str) -> Tuple[Optional[str], Optional[int]]:
    t = normalize(text)

    train = None
    m = RE_TRAIN.search(t)
    if m:
        train = f"T{m.group(1)}"

    car = None
    m2 = RE_CAR.search(t)
    if m2:
        for g in m2.groups():
            if g and g.isdigit():
                v = int(g)
                if 1 <= v <= 99:
                    car = v
                    break

    return train, car


# -------------------------
# Aggression & Flood
# -------------------------

_SWEAR = {
    "тупой", "идиот", "дебил", "сука", "бляд", "нахуй", "хуй", "пизд",
    "fuck", "shit"
}


def _aggression_score(text: str) -> int:
    t = normalize(text)
    toks = set(_tokens(t))
    sc = 0
    if any(w in t for w in _SWEAR) or any(w in toks for w in _SWEAR):
        sc += 2
    if text and sum(1 for c in text if c.isupper()) >= 10:
        sc += 1
    if "!!!" in (text or ""):
        sc += 1
    return sc


def detect_aggression_and_flood(
    session: Dict[str, Any],
    text: str,
    now: Optional[float] = None,
) -> Tuple[int, bool, bool]:
    """
    Обновляет session["aggression"] и session["flood"].
    Возвращает: (aggression_level, is_angry, is_flood)
    """
    if now is None:
        now = time.time()

    # aggression
    aggr = int(session.get("aggression", 0))
    aggr = max(0, aggr - 1)  # небольшой decay
    aggr += _aggression_score(text)
    session["aggression"] = aggr
    is_angry = aggr >= 2

    # flood
    flood = session.get("flood") or {"window_start": now, "count": 0}
    ws = float(flood.get("window_start", now))
    if now - ws > getattr(settings, "FLOOD_WINDOW_SEC", 8):
        flood["window_start"] = now
        flood["count"] = 0
    flood["count"] = int(flood.get("count", 0)) + 1
    session["flood"] = flood

    max_msg = getattr(settings, "FLOOD_MAX_MSG", 4)
    is_flood = int(flood["count"]) >= max_msg

    return aggr, is_angry, is_flood


# -------------------------
# Rule-based NLU object
# -------------------------

_GREETINGS = {
    "здравствуйте", "здрасти", "привет", "салам", "сәлем",
    "hello", "hi", "hey",
    "добрый день", "доброе утро", "добрый вечер",
}

_THANK = {"спасибо", "благодарю", "благодарность", "рахмет", "thx", "thanks"}
_LOST = {"потерял", "потеряла", "забыл", "забыла", "оставил", "оставила", "забыли", "оставили", "пропал", "пропала"}
_DELAY = {"опоздал", "опоздание", "задержка", "задержали", "задержался", "час", "минут", "мин"}
_COMPLAINT = {"жалоба", "плохо", "ужас", "хам", "хамство", "груб", "гряз", "слом", "не работает"}
_CANCEL = {"отмена", "отменить", "закройте", "закрыть", "не актуально", "неактуально", "нашлась", "нашел", "нашёл", "нашла", "нашли"}


def _is_greeting_only(text: str) -> bool:
    t = normalize(text)
    if not t:
        return False
    t = re.sub(r"[^\w\s]", " ", t)
    t = " ".join(t.split())
    if t in _GREETINGS:
        return True
    toks = _tokens(t)
    if len(toks) <= 2 and any("привет" in x or "здрав" in x or x in ("hi", "hello") for x in toks):
        return True
    return False


def _meaning_score(text: str) -> int:
    t = normalize(text)
    toks = set(_tokens(t))
    sc = 0

    if any(w in toks for w in _THANK):
        sc += 2
    if any(w in toks for w in _LOST):
        sc += 2
    if any(w in toks for w in _DELAY):
        sc += 2
    if any(w in toks for w in _COMPLAINT):
        sc += 1

    tr, car = extract_train_and_car(t)
    if tr:
        sc += 2
    if car is not None:
        sc += 2

    if len(_tokens(t)) <= 3 and RE_NUM.search(t):
        sc += 1

    return sc


def _detect_intents(text: str) -> List[str]:
    t = normalize(text)
    toks = set(_tokens(t))

    has_thanks = any(w in toks for w in _THANK)
    has_lost = any(w in toks for w in _LOST)
    has_delay = any(w in toks for w in _DELAY)
    has_compl = any(w in toks for w in _COMPLAINT) or "жалоб" in t

    # Если чистая благодарность — НЕ делаем жалобу
    if has_thanks and not (has_lost or has_delay or has_compl):
        return ["gratitude"]

    intents: List[str] = []
    if has_lost:
        intents.append("lost")
    if has_delay or has_compl:
        intents.append("complaint")
    if has_thanks:
        intents.append("gratitude")

    # уникальность
    out = []
    for i in intents:
        if i not in out:
            out.append(i)
    return out


def _is_cancel(text: str) -> bool:
    t = normalize(text)
    return any(w in t for w in _CANCEL)


@dataclass
class NLUResult:
    greeting_only: bool
    cancel: bool
    meaning_score: int
    intents: List[str]
    slots: Dict[str, Any]


class RuleNLU:
    def analyze(self, text: str) -> NLUResult:
        greeting = _is_greeting_only(text)
        cancel = _is_cancel(text)
        ms = _meaning_score(text)
        intents = _detect_intents(text)
        tr, car = extract_train_and_car(text)

        slots: Dict[str, Any] = {}
        if tr:
            slots["train"] = tr
        if car is not None:
            slots["car"] = car

        return NLUResult(
            greeting_only=greeting,
            cancel=cancel,
            meaning_score=ms,
            intents=intents,
            slots=slots,
        )


def build_nlu() -> RuleNLU:
    return RuleNLU()

