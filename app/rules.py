from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re


GREETINGS = {
    "здравствуйте", "здрасти", "привет", "салам", "сәлем", "hello", "hi", "hey",
    "добрый день", "доброе утро", "добрый вечер",
}

THANK_WORDS = {"спасибо", "благодарю", "благодарность", "рахмет", "thx", "thanks"}
LOST_WORDS = {"потерял", "потеряла", "забыл", "забыла", "оставил", "оставила", "забыли", "оставили", "пропала", "пропал"}
ITEM_WORDS = {"сумка", "рюкзак", "пакет", "кошелек", "кошелёк", "телефон", "паспорт", "наушники", "чемодан", "портмоне"}
DELAY_WORDS = {"опоздал", "опоздание", "задержка", "задержали", "задержался", "час", "минут", "мин", "поздно"}
COMPLAINT_WORDS = {"жалоба", "плохо", "ужас", "хам", "хамство", "груб", "гряз", "не работает", "нет", "слом", "не дали", "не пустили"}
CANCEL_WORDS = {"отмена", "отменить", "закройте", "закрыть", "не актуально", "неактуально", "всё ок", "все ок", "нашлась", "нашёл", "нашел", "нашла", "нашли"}

SWEAR_WORDS = {
    "тупой", "идиот", "дебил", "сука", "бляд", "нахуй", "хуй", "пизд", "fuck", "shit",
}

RE_TRAIN = re.compile(r"(?i)\b[тt]\s*[-]?\s*(\d{1,4})\b")
RE_CAR = re.compile(r"(?i)\b(\d{1,2})\s*(вагон|вгн|ваг)\b|\bвагон\s*(\d{1,2})\b")
RE_NUM = re.compile(r"\b(\d{1,4})\b")
RE_PLACE = re.compile(r"(?i)\b(место|seat)\s*#?\s*(\d{1,3})\b|\b(\d{1,3})\s*(место)\b")
RE_COMPART = re.compile(r"(?i)\b(купе)\s*(\d{1,2})\b|\b(\d{1,2})\s*(купе)\b")


def normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("ё", "е")
    return t


def tokens(text: str) -> List[str]:
    t = normalize(text)
    return re.findall(r"[a-zа-я0-9]+", t)


def is_greeting_only(text: str) -> bool:
    t = normalize(text)
    if not t:
        return False
    t = re.sub(r"[^\w\s]", " ", t)
    t = " ".join(t.split())
    if t in GREETINGS:
        return True
    # короткие "привет" + смайлики
    toks = tokens(t)
    if len(toks) <= 2 and any("привет" in x or "здрав" in x or x in ("hi", "hello") for x in toks):
        return True
    return False


def meaning_score(text: str) -> int:
    t = normalize(text)
    sc = 0
    toks = set(tokens(t))

    if any(w in toks for w in THANK_WORDS):
        sc += 2
    if any(w in toks for w in LOST_WORDS) or any(w in t for w in ITEM_WORDS):
        sc += 2
    if any(w in toks for w in DELAY_WORDS):
        sc += 2
    if any(w in toks for w in COMPLAINT_WORDS):
        sc += 1
    if RE_TRAIN.search(t):
        sc += 2
    if RE_CAR.search(t) or ("вагон" in t and RE_NUM.search(t)):
        sc += 2
    if RE_PLACE.search(t) or RE_COMPART.search(t):
        sc += 1

    # просто числа тоже могут быть смыслом, если pending (это учтётся позже),
    # но для "смысл есть/нет" добавим чуть-чуть:
    if len(tokens(t)) <= 3 and RE_NUM.search(t):
        sc += 1

    return sc


def detect_aggression(text: str) -> int:
    t = normalize(text)
    toks = set(tokens(t))
    sc = 0
    if any(w in t for w in SWEAR_WORDS) or any(w in toks for w in SWEAR_WORDS):
        sc += 2
    if text and sum(1 for c in text if c.isupper()) >= 10:
        sc += 1
    if "!!!" in (text or ""):
        sc += 1
    return sc


def parse_train(text: str) -> Optional[str]:
    m = RE_TRAIN.search(normalize(text))
    if not m:
        return None
    num = m.group(1)
    return f"T{num}"


def parse_car(text: str) -> Optional[int]:
    t = normalize(text)
    m = RE_CAR.search(t)
    if m:
        # варианты групп
        for g in m.groups():
            if g and g.isdigit():
                v = int(g)
                if 1 <= v <= 99:
                    return v
    # если "7 вагон" не совпало, попробуем отдельный вариант: первая цифра если рядом "вагон"
    if "вагон" in t:
        m2 = RE_NUM.search(t)
        if m2:
            v = int(m2.group(1))
            if 1 <= v <= 99:
                return v
    return None


def parse_place(text: str) -> Optional[str]:
    t = normalize(text)
    parts: List[str] = []

    m = RE_PLACE.search(t)
    if m:
        seat = m.group(2) or m.group(3)
        if seat:
            parts.append(f"место {seat}")

    m2 = RE_COMPART.search(t)
    if m2:
        comp = m2.group(2) or m2.group(3)
        if comp:
            parts.append(f"купе {comp}")

    if "верх" in t:
        parts.append("верхняя полка")
    if "ниж" in t:
        parts.append("нижняя полка")
    if "тамбур" in t:
        parts.append("тамбур")
    if "коридор" in t:
        parts.append("коридор")

    if not parts:
        return None

    return ", ".join(dict.fromkeys(parts))


def parse_short_number(text: str) -> Optional[int]:
    """Для pending-slot: '8', '8 говорю же' -> 8"""
    t = normalize(text)
    m = RE_NUM.search(t)
    if not m:
        return None
    v = int(m.group(1))
    return v


def detect_intents(text: str) -> List[str]:
    """
    Возвращает список intent-ов:
    - complaint
    - lost
    - gratitude
    """
    t = normalize(text)
    toks = set(tokens(t))

    has_grat = any(w in toks for w in THANK_WORDS)
    has_lost = any(w in toks for w in LOST_WORDS) or any(w in t for w in ITEM_WORDS) or "потерял" in t
    has_delay = any(w in toks for w in DELAY_WORDS) or "опоздал" in t
    has_compl = any(w in toks for w in COMPLAINT_WORDS) or "жалоб" in t

    intents: List[str] = []

    # приоритет: если человек пишет "благодарность" — это не жалоба
    if has_grat and not (has_lost or has_delay or has_compl):
        return ["gratitude"]

    # мульти-интент
    if has_lost:
        intents.append("lost")

    if has_delay or has_compl:
        intents.append("complaint")

    if has_grat and "gratitude" not in intents:
        intents.append("gratitude")

    # если ничего не нашли — пусто
    return intents


def detect_cancel(text: str) -> bool:
    t = normalize(text)
    return any(w in t for w in CANCEL_WORDS)


def extract_slots(text: str) -> Dict[str, object]:
    """Достаём train/car/place в любом сообщении."""
    out: Dict[str, object] = {}
    tr = parse_train(text)
    if tr:
        out["train"] = tr
    car = parse_car(text)
    if car is not None:
        out["car"] = car
    place = parse_place(text)
    if place:
        out["place"] = place
    return out
