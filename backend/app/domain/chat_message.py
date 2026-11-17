"""
=============================================================================
☕ DOMAIN ENTITY - CHAT REQUEST & RESPONSE
=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ChatRequest:
    message: str


@dataclass
class ChatResponse:
    reply: str
    intent: str
    mode: str = "rag_engine"
    items: List[Dict[str, Any]] = field(default_factory=list)
    tables_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "intent": self.intent,
            "mode": self.mode,
            "items": self.items,
            "tables_info": self.tables_info
        }
