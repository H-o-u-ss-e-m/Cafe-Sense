"""
=============================================================================
☕ DOMAIN ENTITY - TABLE STATE
=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TableEntity:
    table_id: int
    status: str  # "LIBRE", "OCCUPEE", "EN DÉPART"
    persons_count: int = 0
    timer_display: str = "00:00:00"
    leaving_countdown_sec: Optional[int] = None
    bbox: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "status": self.status,
            "persons_count": self.persons_count,
            "timer_display": self.timer_display,
            "leaving_countdown_sec": self.leaving_countdown_sec,
            "bbox": self.bbox
        }


@dataclass
class CafeRoomState:
    total_tables: int = 0
    free_count: int = 0
    occupied_count: int = 0
    leaving_count: int = 0
    tables: List[TableEntity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tables": self.total_tables,
            "free_count": self.free_count,
            "occupied_count": self.occupied_count,
            "leaving_count": self.leaving_count,
            "tables": [t.to_dict() if isinstance(t, TableEntity) else t for t in self.tables]
        }
