"""
=============================================================================
☕ REPOSITORY - MENU REPOSITORY
=============================================================================
Responsable de l'accès à la source de données menu.json.
Fournit les opérations de lecture et filtrage.
=============================================================================
"""

import json
import os
from typing import List, Dict, Any, Optional
from ..core.config import Config
from ..domain.menu_item import MenuItemEntity, FormulaEntity


class MenuRepository:
    def __init__(self, json_path: Optional[str] = None):
        self.json_path = json_path or Config.MENU_JSON_PATH
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Menu JSON introuvable : {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def reload(self):
        self._data = self._load()

    def get_cafe_info(self) -> Dict[str, Any]:
        return self._data.get("cafe", {})

    def get_categories(self) -> List[Dict[str, Any]]:
        return self._data.get("categories", [])

    def get_all_items(self) -> List[MenuItemEntity]:
        items_raw = self._data.get("items", [])
        return [
            MenuItemEntity(
                id=item["id"],
                name=item["name"],
                category=item["category"],
                price=item["price"],
                description=item["description"],
                ingredients=item.get("ingredients", []),
                allergens=item.get("allergens", []),
                tags=item.get("tags", []),
                calories=item.get("calories", 0),
                available=item.get("available", True),
                image=item.get("image"),
                aliases=item.get("aliases", [])
            )
            for item in items_raw
        ]

    def get_item_by_id(self, item_id: int) -> Optional[MenuItemEntity]:
        for it in self.get_all_items():
            if it.id == item_id:
                return it
        return None

    def get_all_formulas(self) -> List[FormulaEntity]:
        formulas_raw = self._data.get("formulas", [])
        return [
            FormulaEntity(
                id=f["id"],
                name=f["name"],
                price=f["price"],
                description=f["description"],
                includes=f.get("includes", []),
                conditions=f.get("conditions", ""),
                persons=f.get("persons", 1),
                available=f.get("available", True)
            )
            for f in formulas_raw
        ]
