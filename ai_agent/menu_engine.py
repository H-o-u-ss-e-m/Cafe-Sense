"""
=============================================================================
☕ MOTEUR DE RECHERCHE ET BASE DE CONNAISSANCES DU MENU DU CAFÉ (V2 Optimisée)
=============================================================================
Fournit les outils de recherche rapide, filtrage multicritères, élimination
des stop-words, scoring sémantique et tolérance aux fautes de frappe.
Aucune API externe requise : 100% autonome et local.
=============================================================================
"""

import json
import os
import re
import unicodedata
from typing import List, Dict, Any, Optional

# Mots vides (stop-words) en français et tunisien à ignorer lors de la recherche
STOP_WORDS = {
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "le", "la", "les",
    "un", "une", "des", "du", "de", "d", "l", "en", "au", "aux", "pour",
    "dans", "sur", "avec", "sans", "est", "sont", "ce", "c", "que", "qui",
    "qu", "ceux", "cette", "cet", "ces", "mon", "ton", "son", "notre",
    "votre", "leur", "mes", "tes", "ses", "nos", "vos", "leurs", "veux",
    "voudrais", "cherche", "donne", "donnez", "donner", "moi", "avez", "avoir",
    "liste", "list", "carte", "menu", "svp", "stp", "merci", "fama",
    "chbiha", "chnouma", "aatini", "a3tini", "andkom", "3andkom", "nheb", "ya",
    "ests", "equil", "estce", "quil", "estceque", "chneya", "chkoun",
    "parle", "proposez", "proposer", "quoi", "quel", "quels", "quelle", "quelles",
    "combien", "coute", "coutent", "prix", "tarif", "tarifs", "soum", "b9adeh", "bkaddeh", "bqadeh",
    "disponible", "disponibles", "dispo", "dispos", "disponibilite"
}

# Corrections des fautes de frappe courantes
TYPO_CORRECTIONS = {
    "jut": "jus",
    "juts": "jus",
    "ju": "jus",
    "donenr": "donner",
    "doner": "donner",
    "desponible": "disponible",
    "desponibles": "disponible",
    "cappucin": "capucin",
    "kapucin": "capucin",
    "capucino": "cappuccino",
    "expres": "express",
    "expreso": "espresso",
    "oja": "ojja",
    "ouja": "ojja",
    "sandwitch": "sandwich",
    "sandwish": "sandwich",
    "frikasse": "fricasse",
    "fricassée": "fricasse",
    "makrouna": "makarouna",
    "lazagne": "lasagne",
    "cheescake": "cheesecake",
    "chocolat": "chocolat"
}


def normalize_text(text: str) -> str:
    """Normalise une chaîne : minuscules, suppression des accents et caractères arabizi."""
    if not text:
        return ""
    text = text.lower().strip()
    # Remplacer les caractères arabes latinisés fréquents
    text = text.replace("3", "a").replace("7", "h").replace("9", "q").replace("5", "kh")
    nfkd = unicodedata.normalize('NFKD', text)
    cleaned = "".join([c for c in nfkd if not unicodedata.combining(c)])
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_keywords(text: str) -> List[str]:
    """Extrait les mots-clés pertinents en éliminant les stop-words et en corrigeant les fautes."""
    normalized = normalize_text(text)
    words = normalized.split()
    keywords = []
    for w in words:
        # Correction typo
        w = TYPO_CORRECTIONS.get(w, w)
        if w not in STOP_WORDS and len(w) > 1:
            keywords.append(w)
    return keywords


class MenuEngine:
    def __init__(self, json_path: Optional[str] = None):
        if json_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "data", "menu.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(os.path.dirname(current_dir), "ai_agent", "data", "menu.json")

        self.json_path = json_path
        self.data = self._load_data()
        self.cafe = self.data.get("cafe", {})
        self.categories = self.data.get("categories", [])
        self.items = self.data.get("items", [])
        self.formulas = self.data.get("formulas", [])
        self.faq = self.data.get("faq", [])

    def _load_data(self) -> Dict[str, Any]:
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Fichier menu introuvable : {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def detect_category(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Détecte si la question demande explicitement une catégorie entière (ex: 'les jus', 'liste des cafés', 'fama sandwich').
        Retourne un dict avec 'id', 'name', 'icon' si c'est une demande de catégorie, ou None si c'est un article précis ou autre chose.
        """
        keywords = extract_keywords(query)
        norm = normalize_text(query)
        words = norm.split()
        corrected_words = [TYPO_CORRECTIONS.get(w, w) for w in words]

        category_defs = {
            "jus": {
                "name": "Jus et Smoothies",
                "icon": "🥤",
                "aliases": ["jus", "smoothie", "smoothies", "cocktail", "cocktails", "juice", "3assir", "assir"]
            },
            "cafe": {
                "name": "Cafés",
                "icon": "☕",
                "aliases": ["cafe", "cafes", "coffee", "coffees", "kahwa", "qahwa"]
            },
            "the": {
                "name": "Thés et Infusions",
                "icon": "🍵",
                "aliases": ["the", "thes", "tea", "infusion", "infusions", "tisane", "tisanes", "atay"]
            },
            "boisson_froide": {
                "name": "Boissons Fraîches",
                "icon": "🧊",
                "aliases": ["boisson", "boissons", "soda", "sodas", "gazouz"]
            },
            "sandwich": {
                "name": "Sandwichs & Paninis",
                "icon": "🥪",
                "aliases": ["sandwich", "sandwichs", "sandwiches", "casse croute", "panini", "paninis"]
            },
            "salade": {
                "name": "Salades",
                "icon": "🥗",
                "aliases": ["salade", "salades", "salad", "slata"]
            },
            "pizza": {
                "name": "Pizzas",
                "icon": "🍕",
                "aliases": ["pizza", "pizzas"]
            },
            "pates": {
                "name": "Pâtes",
                "icon": "🍝",
                "aliases": ["pates", "pasta", "spaghetti", "penne", "tagliatelle", "lasagne", "makarouna"]
            },
            "plat": {
                "name": "Plats",
                "icon": "🍽️",
                "aliases": ["plat", "plats", "repas"]
            },
            "dessert": {
                "name": "Desserts & Pâtisseries",
                "icon": "🍰",
                "aliases": ["dessert", "desserts", "gateau", "gateaux", "cake", "douceur", "7lou"]
            },
            "viennoiserie": {
                "name": "Viennoiseries",
                "icon": "🥐",
                "aliases": ["viennoiserie", "viennoiseries"]
            },
            "petit_dejeuner": {
                "name": "Petits-Déjeuners",
                "icon": "🍳",
                "aliases": ["petit dejeuner", "breakfast", "ftour", "brunch"]
            }
        }

        # Mots indiquant une intention de liste globale
        list_intent_words = {"liste", "list", "quels", "quelles", "quel", "quelle", "tous", "toutes", 
                             "proposez", "proposer", "choix", "variete", "options", "carte", "menu", "types", "sortes"}
        has_list_intent = any(w in corrected_words for w in list_intent_words)

        for cat_id, info in category_defs.items():
            matched_alias = None
            for alias in info["aliases"]:
                if alias in corrected_words or alias in norm:
                    matched_alias = alias
                    break

            if matched_alias:
                # Vérifier si les autres mots-clés contiennent un article spécifique (ex: 'orange', 'thon', 'fraise', 'vanille')
                alias_words = matched_alias.split()
                non_cat_keywords = [kw for kw in keywords if kw not in alias_words and kw not in list_intent_words]

                # S'il y a une intention de liste explicite (ex: "liste des jus", "quels sont les cafés")
                # OU si la requête ne contient que le nom de la catégorie (ex: "les jus", "fama sandwich", "jus")
                if has_list_intent or len(non_cat_keywords) == 0:
                    return {
                        "id": cat_id,
                        "name": info["name"],
                        "icon": info["icon"]
                    }

        return None

    def search_items(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        tag: Optional[str] = None,
        exclude_allergens: Optional[List[str]] = None,
        available_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Recherche multicritère avec calcul de score de pertinence et tri décroissant."""
        keywords = extract_keywords(query) if query else []

        scored_results = []
        for item in self.items:
            if available_only and not item.get("available", True):
                continue

            # Filtre par prix
            price = item.get("price", 0.0)
            if max_price is not None and price > max_price:
                continue
            if min_price is not None and price < min_price:
                continue

            # Filtre par catégorie stricte si spécifiée
            if category:
                if item.get("category", "").lower() != category.lower():
                    continue

            # Filtre par tag
            if tag:
                item_tags = [t.lower() for t in item.get("tags", [])]
                if tag.lower() not in item_tags:
                    continue

            # Filtre par allergène
            if exclude_allergens:
                item_allergens = [a.lower() for a in item.get("allergens", [])]
                if any(ex.lower() in item_allergens for ex in exclude_allergens):
                    continue

            # Calcul du score de pertinence si des mots-clés sont fournis
            score = 0
            if keywords:
                item_name_norm = normalize_text(item["name"])
                item_cat_norm = normalize_text(item["category"])
                item_desc_norm = normalize_text(item["description"])
                item_aliases_norm = " ".join([normalize_text(a) for a in item.get("aliases", [])])
                item_ings_norm = " ".join([normalize_text(i) for i in item.get("ingredients", [])])

                for kw in keywords:
                    # 1. Correspondance exacte dans le nom : +100 points
                    if kw in item_name_norm.split():
                        score += 100
                    elif kw in item_name_norm:
                        score += 60

                    # 2. Correspondance dans les alias : +80 points
                    if kw in item_aliases_norm.split():
                        score += 80
                    elif kw in item_aliases_norm:
                        score += 40

                    # 3. Correspondance dans la catégorie : +50 points
                    if kw == item_cat_norm:
                        score += 50

                    # 4. Ingrédients : +25 points
                    if kw in item_ings_norm.split():
                        score += 25

                    # 5. Description : +10 points
                    if kw in item_desc_norm.split():
                        score += 10

                if score == 0:
                    continue  # Aucun mot-clé pertinent ne correspond
            else:
                score = 1  # Pas de mot-clé, on conserve tout

            scored_results.append((score, item))

        # Trier par score décroissant (les plus pertinents en premier)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_results]

    def find_item_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Trouve un article spécifique avec score élevé."""
        if not name:
            return None
        norm_name = normalize_text(name)
        keywords = extract_keywords(name)
        if not keywords:
            return None

        # Si les mots-clés sont uniquement un nom générique de catégorie (ex: ['jus'], ['cafe']), ce n'est pas un article précis
        GENERIC_CATEGORY_WORDS = {"jus", "smoothie", "cafe", "cafes", "the", "sandwich", "sandwichs", "pizza", "pizzas", "salade", "salades", "dessert", "desserts", "plat", "plats", "pates"}
        if len(keywords) == 1 and keywords[0] in GENERIC_CATEGORY_WORDS:
            return None

        # 1. Correspondance exacte sur le nom
        for item in self.items:
            if normalize_text(item["name"]) == norm_name:
                return item

        # 2. Correspondance exacte dans les alias
        for item in self.items:
            for alias in item.get("aliases", []):
                if normalize_text(alias) == norm_name:
                    return item

        # 3. Recherche pondérée
        scored = self.search_items(query=name)
        if scored:
            # Si le premier résultat a un nom qui contient tous les mots clés
            top = scored[0]
            top_name_norm = normalize_text(top["name"])
            top_aliases = " ".join([normalize_text(a) for a in top.get("aliases", [])])
            if all(kw in top_name_norm or kw in top_aliases for kw in keywords):
                return top

        return None

    def get_formulas(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère les formules du menu."""
        if not query:
            return self.formulas
        norm = normalize_text(query)
        res = []
        for f in self.formulas:
            haystack = normalize_text(f["name"]) + " " + normalize_text(f.get("description", ""))
            if any(w in haystack for w in norm.split() if w not in STOP_WORDS):
                res.append(f)
        return res if res else self.formulas
