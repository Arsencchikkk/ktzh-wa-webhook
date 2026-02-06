from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Almaty")

# ===== Regex =====
TRAIN_RE = re.compile(r"\b(?:поезд|пойыз|train)?\s*([TТ]\s*\d{1,4}|\d{2,4}[A-Za-zА-Яа-яЁё])\b", re.IGNORECASE)
WAGON_RE_1 = re.compile(r"(?:вагон|вагоне|вагонда|wagon)\s*[:№]?\s*(\d{1,2})", re.IGNORECASE)
WAGON_RE_2 = re.compile(r"\b(\d{1,2})\s*(?:вагон|вагоне|вагонда)\b", re.IGNORECASE)
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")
DATE_RE = re.compile(r"\b(\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)\b")

# route like "Семей-Кызылорда", allow 1-3 tokens per side
LET = r"A-Za-zА-Яа-яЁёҚқҮүҰұӨөӘәІіҢңҒғҺһ"
ROUTE_RE = re.compile(
    rf"([{LET}]+(?:\s+[{LET}]+){{0,2}})\s*[-–—]\s*([{LET}]+(?:\s+[{LET}]+){{0,2}})",
    re.IGNORECASE
)

# ===== Keywords =====
GREET_RX = re.compile(r"^(здравствуй(те)?|сәлем(етсіздер ме|етсіз бе)?|саламат(сыздарма|сыз ба)?|привет)\W*$", re.IGNORECASE)
THANKS_RX = re.compile(r"(спасибо|благодар(ю|ность)|рахмет|үлкен рахмет)", re.IGNORECASE)
LOST_RX = re.compile(r"(забыл|забыли|оставил|оставила|потерял|потеряла|пропал|пропала|термос|сумк|кошелек|қалды|ұмытып|жоғал)", re.IGNORECASE)
FOUND_RX = re.compile(r"(наш(ел|лась|лись)|табылды|found)", re.IGNORECASE)

COMPLAINT_RX = re.compile(r"(жалоб|плохо|ужас|не работает|нет\s+|жарко|холодно|гряз|туалет|опозд|задерж|хам|грубо|проводник)", re.IGNORECASE)

INFO_Q_RX = re.compile(r"(\?|как\b|где\b|сколько\b|қалай\b|қайдан\b|қай жерде\b)", re.IGNORECASE)

CAT_TEMPERATURE = re.compile(r"(жарко|холодно|температур|кондиционер|печк|отоплен)", re.IGNORECASE)
CAT_SANITARY = re.compile(r"(туалет|бумаг|гряз|санитар|запах|вода\s+нет|нет\s+воды)", re.IGNORECASE)
CAT_SERVICE = re.compile(r"(хам|грубо|не помог|сервис|обслужив)", re.IGNORECASE)
CAT_DELAY = re.compile(r"(опозд|задержк|кешікт|кешіг)", re.IGNORECASE)
CAT_CONDUCTOR = re.compile(r"(проводник|начальник поезда|жолсерік)", re.IGNORECASE)

# Severity keywords (very rough MVP)
SEV5 = re.compile(r"(пожар|дым|угроза|опасн|драка|плохо стало|скорую|кровь|травм)", re.IGNORECASE)
SEV4 = re.compile(r"(антисанитар|нет\s+туалет|туалет\s+не\s+работает|очень\s+жарко|очень\s+холодно|невозможно)", re.IGNORECASE)

@dataclass
class NLUResult:
    intent: str                 # greeting|gratitude|complaint|lost_and_found|info|other
    language: str               # ru|kk|mixed
    slots: Dict[str, Any]
    categories: List[str]
    severity: Dict[str, Any]
    is_found_message: bool

def detect_language(text: str) -> str:
    # simple heuristic for Kazakh letters
    kk_letters = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")
    has_kk = any(ch in kk_letters for ch in text)
    has_ru = any("А" <= ch <= "я" or ch in "Ёё" for ch in text)
    if has_kk and has_ru:
        return "mixed"
    if has_kk:
        return "kk"
    if has_ru:
        return "ru"
    return "ru"

def _parse_relative_date(text: str) -> Optional[str]:
    t = text.lower()
    now = datetime.now(TZ).date()
    if "сегодня" in t or "бүгін" in t:
        return now.isoformat()
    if "вчера" in t or "кеше" in t:
        return (now - timedelta(days=1)).isoformat()
    if "завтра" in t or "ертең" in t:
        return (now + timedelta(days=1)).isoformat()
    return None

def extract_slots(text: str) -> Dict[str, Any]:
    t = text.strip()

    train = None
    m = TRAIN_RE.search(t)
    if m:
        train = m.group(1).upper().replace(" ", "")

    wagon = None
    m = WAGON_RE_1.search(t) or WAGON_RE_2.search(t)
    if m:
        try:
            wagon = int(m.group(1))
        except Exception:
            wagon = None

    time = None
    m = TIME_RE.search(t)
    if m:
        time = m.group(0)

    date = _parse_relative_date(t)
    if not date:
        m = DATE_RE.search(t)
        if m:
            date = m.group(1)

    route_from = route_to = None
    m = ROUTE_RE.search(t)
    if m:
        route_from = m.group(1).strip()
        route_to = m.group(2).strip()

    # item description (very simple)
    item = None
    if LOST_RX.search(t):
        # take whole text as fallback
        item = t

    return {
        "train": train,
        "wagon": wagon,
        "time": time,
        "date": date,
        "routeFrom": route_from,
        "routeTo": route_to,
        "item": item,
    }

def detect_categories(text: str) -> List[str]:
    cats = []
    if CAT_TEMPERATURE.search(text):
        cats.append("температура")
    if CAT_SANITARY.search(text):
        cats.append("санитария")
    if CAT_DELAY.search(text):
        cats.append("опоздание")
    if CAT_SERVICE.search(text):
        cats.append("сервис")
    if CAT_CONDUCTOR.search(text):
        cats.append("проводник")
    return list(dict.fromkeys(cats))

def detect_severity(text: str) -> Dict[str, Any]:
    if SEV5.search(text):
        return {"score": 5, "reason": "ключевые слова безопасности/здоровья"}
    if SEV4.search(text):
        return {"score": 4, "reason": "сильная проблема комфорта/санитарии"}
    if COMPLAINT_RX.search(text):
        return {"score": 3, "reason": "обычная жалоба"}
    return {"score": 1, "reason": "не жалоба или нейтрально"}

def detect_intent(text: str) -> str:
    t = text.strip()
    if not t:
        return "other"
    if GREET_RX.match(t):
        return "greeting"
    if THANKS_RX.search(t):
        # gratitude may co-exist with complaint, but for MVP treat as gratitude if no complaint keyword
        if COMPLAINT_RX.search(t):
            return "complaint"
        return "gratitude"
    if FOUND_RX.search(t) and not COMPLAINT_RX.search(t):
        return "other"
    if LOST_RX.search(t):
        return "lost_and_found"
    if COMPLAINT_RX.search(t):
        return "complaint"
    if INFO_Q_RX.search(t):
        return "info"
    return "other"

def run_nlu(text: str) -> NLUResult:
    lang = detect_language(text)
    intent = detect_intent(text)
    slots = extract_slots(text)
    cats = detect_categories(text)
    sev = detect_severity(text) if intent == "complaint" else {"score": 1, "reason": "not_complaint"}
    is_found = bool(FOUND_RX.search(text))
    return NLUResult(intent=intent, language=lang, slots=slots, categories=cats, severity=sev, is_found_message=is_found)
