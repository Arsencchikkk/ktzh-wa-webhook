from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
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

RESET_WORDS = {"сброс", "reset", "отмена", "стоп", "начать заново"}  # (для диалога)

GRATITUDE_WORDS = {
    "благодарность", "спасибо", "рахмет", "алғыс", "хочу поблагодарить", "благодарю",
    "выражаю благодарность"
}

LOST_WORDS = {
    "потерял", "потеряла", "забыл", "забыла", "оставил", "оставила", "пропала", "пропал",
    "сумка", "рюкзак", "кошелек", "кошелёк", "телефон", "документы", "паспорт", "карта", "ноутбук", "чемодан"
}

COMPLAINT_WORDS = {
    "жалоба", "плохо", "ужас", "не работает", "грязно", "хамство", "грубость",
    "опоздал", "опоздание", "задержка", "задержали", "сломано", "холодно", "жарко",
    "не дали", "не пустили", "проводник", "контролер", "контролёр"
}

DELAY_WORDS = {"опоздал", "опоздание", "задержка", "задержали", "задержался", "прибыл поздно"}

# --- auto close intents ---
CANCEL_CASE_WORDS = {
    "отменяю", "отмена", "отменить", "закройте", "закрыть", "не актуально", "уже не нужно",
    "всё решено", "все решено", "разобрались", "можно закрыть", "стоп заявка"
}
FOUND_WORDS = {
    "нашлась", "нашелась", "нашёлся", "нашелся", "нашли", "нашёл", "нашел", "нашла",
    "вернули", "вещь нашлась", "сумка нашлась", "телефон нашелся"
}

# --- anti-flood / aggression (simple rules) ---
PROFANITY_RE = re.compile(
    r"(?i)\b("
    r"бля|бляд|сука|хуй|хуе|пизд|еба|ёба|ебл|нах|мудак|гандон|долбоеб|идиот|тупой"
    r")\b"
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def contains_any(text: str, words: set[str]) -> bool:
    t = normalize(text)
    return any(w in t for w in words)


def extract_train(text: str) -> Optional[str]:
    m = TRAIN_RE.search(text or "")
    return f"T{m.group(1)}" if m else None


def extract_car(text: str) -> Optional[int]:
    txt = text or ""
    for rx in (CAR_RE_1, CAR_RE_2):
        m = rx.search(txt)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 99:
                return v
    m = JUST_INT_RE.match(txt.strip())
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
    t2 = re.sub(r"[^a-zа-яёқңүұһәі ]+", " ", t)
    t2 = normalize(t2)
    return any(t2 == g or t2.startswith(g + " ") or t2.endswith(" " + g) for g in GREETINGS)


def detect_cancel_found(text: str) -> tuple[bool, bool]:
    t = normalize(text)
    is_cancel = any(w in t for w in CANCEL_CASE_WORDS)
    is_found = any(w in t for w in FOUND_WORDS)
    return is_cancel, is_found


def detect_aggression_and_flood(text: str, prev_text: Optional[str], repeat_count: int) -> dict:
    raw = (text or "").strip()
    t = normalize(raw)

    prof = bool(PROFANITY_RE.search(raw))
    excls = raw.count("!") >= 3
    caps_ratio = 0.0
    letters = [c for c in raw if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    caps = caps_ratio >= 0.6 and len(raw) >= 8

    same_as_prev = prev_text and normalize(prev_text) == t and len(t) > 0
    new_repeat = (repeat_count + 1) if same_as_prev else 0
    flooding = new_repeat >= 2  # 3 одинаковых подряд

    angry = prof or excls or caps or flooding
    tone = "angry" if angry else ("positive" if contains_any(t, GRATITUDE_WORDS) else "neutral")

    return {
        "tone": tone,
        "angry": angry,
        "flooding": flooding,
        "repeat_count": new_repeat,
        "prev_text": raw,
    }


@dataclass
class NLUResult:
    intents: List[str]
    train: Optional[str] = None
    carNumber: Optional[int] = None
    complaintText: Optional[str] = None
    gratitudeText: Optional[str] = None
    lostHint: Optional[str] = None
    meaningScore_0_100: int = 0
    isGreetingOnly: bool = False
    isReset: bool = False
    isCancel: bool = False
    isFound: bool = False


def meaning_score(text: str, pending_slots: List[str]) -> int:
    raw = (text or "").strip()
    if not raw:
        return 0

    if pending_slots:
        if extract_train(raw) or extract_car(raw):
            return 90
        if len(normalize(raw)) <= 12:
            return 70

    if is_greeting_only(raw):
        return 0

    if contains_any(raw, GRATITUDE_WORDS | LOST_WORDS | COMPLAINT_WORDS):
        return 90

    if extract_train(raw) or extract_car(raw):
        return 75

    t = normalize(raw)
    if len(t) <= 2:
        return 5
    if len(t) <= 5 and t in {"ок", "окей", "да", "нет", "ага"}:
        return 10
    if len(t) >= 15:
        return 60
    return 35


def classify_intents(text: str) -> List[str]:
    t = normalize(text)
    intents: List[str] = []

    if contains_any(t, GRATITUDE_WORDS):
        intents.append("gratitude")

    if contains_any(t, LOST_WORDS) or "оставил в поезде" in t or "потерял" in t or "забыл" in t:
        intents.append("lost_and_found")

    if contains_any(t, COMPLAINT_WORDS) or contains_any(t, DELAY_WORDS):
        intents.append("complaint")

    out, seen = [], set()
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

    is_cancel, is_found = detect_cancel_found(t)

    train, car = extract_train_and_car(t)
    intents = [] if greet else classify_intents(t)

    res = NLUResult(
        intents=intents,
        train=train,
        carNumber=car,
        meaningScore_0_100=ms,
        isGreetingOnly=greet,
        isReset=is_reset,
        isCancel=is_cancel,
        isFound=is_found,
    )

    if "complaint" in intents:
        res.complaintText = t
    if "gratitude" in intents:
        res.gratitudeText = t
    if "lost_and_found" in intents:
        res.lostHint = t

    return res
