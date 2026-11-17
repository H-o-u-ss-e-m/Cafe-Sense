"""
=============================================================================
☕ SERVICE - MENU SERVICE
=============================================================================
Use cases liés à la consultation du menu, filtres par régime et recherche.
=============================================================================
"""

from typing import List, Dict, Any, Optional
from ..repositories.menu_repository import MenuRepository
from ..domain.menu_item import MenuItemEntity, FormulaEntity


class MenuService:
    def __init__(self, repository: Optional[MenuRepository] = None):
        self.repository = repository or MenuRepository()

    def get_full_menu(self) -> Dict[str, Any]:
        return {
            "cafe": self.repository.get_cafe_info(),
            "categories": self.repository.get_categories(),
            "items": [it.to_dict() for it in self.repository.get_all_items()],
            "formulas": [f.to_dict() for f in self.repository.get_all_formulas()]
        }

    def get_categories(self) -> List[Dict[str, Any]]:
        return self.repository.get_categories()

    def get_items_by_category(self, category: str) -> List[Dict[str, Any]]:
        all_items = self.repository.get_all_items()
        return [it.to_dict() for it in all_items if it.category.lower() == category.lower()]

    def get_formulas(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.repository.get_all_formulas()]

    def filter_items(
        self,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        tag: Optional[str] = None,
        exclude_allergen: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        results = []
        for it in self.repository.get_all_items():
            if not it.available:
                continue
            if category and it.category.lower() != category.lower():
                continue
            if max_price is not None and it.price > max_price:
                continue
            if tag and tag.lower() not in [t.lower() for t in it.tags]:
                continue
            if exclude_allergen and exclude_allergen.lower() in [a.lower() for a in it.allergens]:
                continue
            results.append(it.to_dict())
        return results
