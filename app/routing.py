from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from . import settings

@dataclass
class RoutingDecision:
    region: str
    target_chat_type: Optional[str]
    target_chat_id: Optional[str]
    reason: str

def _match_rule(rule: Dict[str, Any], *, train: Optional[str], route_from: Optional[str], route_to: Optional[str]) -> bool:
    m = rule.get("match", {})
    train_rx = m.get("train_regex")
    if train_rx and train:
        if re.search(train_rx, train, re.IGNORECASE):
            return True

    contains = m.get("route_contains")
    if contains and (route_from or route_to):
        rf = (route_from or "").lower()
        rt = (route_to or "").lower()
        for token in contains:
            tok = str(token).lower()
            if tok in rf or tok in rt:
                return True

    return False

def resolve_region(train: Optional[str], route_from: Optional[str], route_to: Optional[str]) -> str:
    for rule in settings.ROUTING_RULES:
        try:
            if _match_rule(rule, train=train, route_from=route_from, route_to=route_to):
                return rule.get("region", "unknown")
        except Exception:
            continue
    return "unknown"

def resolve_executor(region: str) -> RoutingDecision:
    rec = settings.EXECUTORS.get(region)
    if not rec:
        return RoutingDecision(region=region, target_chat_type=None, target_chat_id=None, reason="no_executor_configured")
    return RoutingDecision(
        region=region,
        target_chat_type=rec.get("chatType"),
        target_chat_id=rec.get("chatId"),
        reason="executor_from_env"
    )
