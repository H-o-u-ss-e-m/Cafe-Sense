"""
=============================================================================
☕ CONTROLLER - MENU ROUTES
=============================================================================
Endpoints REST pour le catalogue du menu, catégories et formules.
=============================================================================
"""

from flask import Blueprint, request, jsonify
from ..services.menu_service import MenuService

menu_bp = Blueprint("menu", __name__)
_menu_service: MenuService = None


def init_menu_routes(menu_service: MenuService):
    global _menu_service
    _menu_service = menu_service


@menu_bp.route("/api/menu")
def api_menu():
    """Retourne l'ensemble du catalogue du menu (catégories, items, formules, café info)."""
    if not _menu_service:
        return jsonify({"error": "Service menu non initialisé"}), 503
    return jsonify(_menu_service.get_full_menu())


@menu_bp.route("/api/menu/categories")
def api_categories():
    """Retourne la liste des catégories du café."""
    if not _menu_service:
        return jsonify({"error": "Service menu non initialisé"}), 503
    return jsonify(_menu_service.get_categories())


@menu_bp.route("/api/menu/filter")
def api_filter():
    """Filtre les articles par catégorie, prix maximum, tag ou allergène."""
    if not _menu_service:
        return jsonify({"error": "Service menu non initialisé"}), 503
    category = request.args.get("category")
    max_price = request.args.get("max_price", type=float)
    tag = request.args.get("tag")
    exclude_allergen = request.args.get("exclude_allergen")
    results = _menu_service.filter_items(category, max_price, tag, exclude_allergen)
    return jsonify(results)


@menu_bp.route("/api/menu/formulas")
def api_formulas():
    """Retourne la liste des formules et promotions."""
    if not _menu_service:
        return jsonify({"error": "Service menu non initialisé"}), 503
    return jsonify(_menu_service.get_formulas())
