"""
=============================================================================
☕ DOMAIN ENTITY - MENU & FORMULAS
=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MenuItemEntity:
    id: int
    name: str
    category: str
    price: float
    description: str
    ingredients: List[str] = field(default_factory=list)
    allergens: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    calories: int = 0
    available: bool = True
    image: Optional[str] = None
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "description": self.description,
            "ingredients": self.ingredients,
            "allergens": self.allergens,
            "tags": self.tags,
            "calories": self.calories,
            "available": self.available,
            "image": self.image,
            "aliases": self.aliases
        }


@dataclass
class FormulaEntity:
    id: int
    name: str
    price: float
    description: str
    includes: List[str] = field(default_factory=list)
    conditions: str = ""
    persons: int = 1
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "description": self.description,
            "includes": self.includes,
            "conditions": self.conditions,
            "persons": self.persons,
            "available": self.available
        }
