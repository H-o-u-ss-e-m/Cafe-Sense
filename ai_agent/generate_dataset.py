"""
=============================================================================
☕ GÉNÉRATEUR DU DATASET D'ENTRAÎNEMENT POUR L'AGENT IA SMART CAFÉ
=============================================================================
Génère un dataset riche et diversifié (en français, dialecte tunisien / derja,
et anglais) pour entraîner le modèle LLM local sous Google Colab.
Couvre :
- Questions de prix et descriptions
- Ingrédients et allergènes (gluten, lactose, arachides, etc.)
- Régimes alimentaires (végétarien, vegan, sans caféine, etc.)
- Formules et promotions (étudiant, duo, matin, midi, famille)
- Informations café (wifi, horaires, parking, réservation, livraison)
- Disponibilité des tables (intégration Vision)
- Formulations familières et dialectales tunisiennes ("9adeh soum", "fama", "nheb")
- Politesse, salutations et demandes hors-sujet
=============================================================================
"""

import json
import os
import random
from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "Tu es l'agent conversationnel intelligent du café-restaurant 'Coffee House Lac' à Tunis. "
    "Ton rôle est d'accueillir chaleureusement les clients, de répondre à leurs questions sur le menu, "
    "les prix en dinars tunisiens (DT), les ingrédients, les allergènes, les formules et les services du café (wifi, horaires, livraison). "
    "Tu t'exprimes avec politesse, clarté et précision. Tu comprends le français, l'anglais et le dialecte tunisien (Derja/arabizi). "
    "Ne donne que des informations exactes basées sur le menu officiel du café."
)


def build_training_dataset(menu_path: str, output_path: str, num_variations: int = 400) -> List[Dict[str, Any]]:
    with open(menu_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    formulas = data.get("formulas", [])
    cafe = data.get("cafe", {})
    categories = data.get("categories", [])
    dataset = []

    def add_dialogue(user_text: str, assistant_text: str):
        dialogue = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text}
            ]
        }
        dataset.append(dialogue)

    # 1. SALUTATIONS & ACCUEIL
    greetings = [
        ("Bonjour !", "Bonjour et bienvenue au Coffee House Lac ! Comment puis-je vous aider aujourd'hui ? Souhaitez-vous découvrir notre menu, nos boissons fraîches ou nos formules ?"),
        ("Salut, vous êtes ouverts ?", f"Bonjour ! Oui, nous sommes ouverts du lundi au vendredi de {cafe['hours_groups'][0]['open']} à {cafe['hours_groups'][0]['close']}, et le week-end de {cafe['hours_groups'][1]['open']} à {cafe['hours_groups'][1]['close']}. Que désirez-vous commander ?"),
        ("Aslema", "3aslema et bienvenue au Coffee House Lac ! Que puis-je vous servir aujourd'hui ?"),
        ("Hi there", "Hello and welcome to Coffee House Lac! How can I assist you today? Would you like to see our menu or special deals?"),
        ("Merci beaucoup", "Je vous en prie ! C'est un plaisir de vous servir. Très bonne dégustation et à très bientôt au Coffee House Lac !"),
        ("3aychek", "Men ghir mziya ! C'est avec grand plaisir. N'hésitez pas si vous avez d'autres questions !")
    ]
    for u, a in greetings:
        add_dialogue(u, a)

    # 2. INFORMATIONS GÉNÉRALES DU CAFÉ
    info_pairs = [
        ("C'est quoi le code wifi ?", "Le Wi-Fi est 100% gratuit pour tous nos clients ! Le code d'accès actuel vous est remis directement à la caisse ou sur votre ticket de caisse."),
        ("Est-ce que le wifi est gratuit ?", "Oui, le Wi-Fi est entièrement gratuit au café. Il vous suffit de demander le mot de passe au comptoir."),
        ("Avez-vous un parking ?", "Oui, nous disposons d'un parking gratuit de 20 places situé juste devant le café pour nos clients."),
        ("Où vous situez-vous exactement ?", f"Nous sommes situés au {cafe['address']}. Vous êtes les bienvenus !"),
        ("Est-ce que vous faites la livraison ?", f"Oui, nous assurons la livraison de {cafe['delivery']['hours']} dans les zones : {', '.join(cafe['delivery']['zones'])}. Frais de livraison : {cafe['delivery']['fee']} DT (commande minimum {cafe['delivery']['min_order']} DT)."),
        ("Faites-vous la livraison à La Marsa ?", "Oui, nous livrons à La Marsa ainsi qu'aux Berges du Lac 1, Lac 2 et Jardins du Lac entre 11h00 et 22h00."),
        ("Comment réserver une table ?", f"Vous pouvez réserver directement par téléphone ou via WhatsApp au {cafe['reservation']['how']}. La réservation est vivement conseillée pour les groupes de 6 personnes ou plus."),
        ("Avez-vous une terrasse pour fumeurs ?", "Oui, nous avons une agréable terrasse extérieure où l'espace fumeurs est autorisé. L'intérieur est climatisé et non-fumeur."),
        ("Acceptez-vous les cartes bancaires ?", "Oui, nous acceptons les paiements en espèces, par carte bancaire et par titres restaurant."),
        ("Est-ce que toute votre nourriture est halal ?", "Oui, l'ensemble de notre cuisine et de nos viandes est certifié 100% halal.")
    ]
    for u, a in info_pairs:
        add_dialogue(u, a)

    # 3. DISPONIBILITÉ DES TABLES (VISION IOT)
    table_pairs = [
        ("Est-ce qu'il y a des tables libres en ce moment ?", "Notre système de vision intelligente surveille l'occupation des tables en direct. Actuellement, des tables sont disponibles en terrasse et en salle. Vous pouvez venir vous installer sans attente !"),
        ("Fama blasa tawwa ?", "Oui, notre caméra intelligente indique qu'il y a des tables libres en ce moment au café. Marhba bik !"),
        ("Y a-t-il beaucoup de monde ?", "Notre moniteur de salle en temps réel affiche plusieurs tables libres prêtes à vous accueillir. N'hésitez pas à venir nous rendre visite !")
    ]
    for u, a in table_pairs:
        add_dialogue(u, a)

    # 4. ARTICLES DU MENU (PRIX, INGRÉDIENTS, ALLERGÈNES)
    price_templates_fr = [
        "Combien coûte {name} ?",
        "Quel est le prix de {name} ?",
        "C'est à combien {name} ?",
        "Je voudrais savoir le prix pour {name}.",
        "Donne-moi le prix de {name} s'il vous plaît."
    ]
    price_templates_tn = [
        "9adeh soum {name} ?",
        "Soum {name} svp ?",
        "B9adeh {name} ?",
        "Fama {name} w b9adeh ?"
    ]
    desc_templates = [
        "Parle-moi de {name}.",
        "Qu'est-ce qu'il y a dans {name} ?",
        "Quels sont les ingrédients de {name} ?",
        "C'est quoi {name} exactement ?"
    ]

    for item in items:
        name = item["name"]
        price = item["price"]
        desc = item["description"]
        ings = ", ".join(item.get("ingredients", []))
        allergens = item.get("allergens", [])

        # Réponse prix
        resp_price = f"Le {name} est à {price:.1f} DT. {desc}"
        resp_price_tn = f"Soum {name} houwa {price:.1f} DT. {desc}"

        # Ajouter variations français
        for tmpl in price_templates_fr[:2]:
            add_dialogue(tmpl.format(name=name), resp_price)

        # Ajouter variation tunisienne
        add_dialogue(price_templates_tn[0].format(name=name), resp_price_tn)

        # Réponse ingrédients
        resp_desc = f"{name} ({price:.1f} DT) contient : {ings}. {desc}"
        add_dialogue(desc_templates[0].format(name=name), resp_desc)

        # Questions allergènes spécifiques
        if allergens:
            all_str = ", ".join(allergens)
            add_dialogue(
                f"Est-ce que {name} contient des allergènes ?",
                f"Attention : {name} contient les allergènes suivants : {all_str}. Si vous avez une allergie alimentaire, prévenez notre équipe lors de la commande."
            )
        else:
            add_dialogue(
                f"Est-ce que {name} contient des allergènes ?",
                f"{name} ne contient aucun des principaux allergènes courants déclarés. Ingrédients : {ings}."
            )

    # 5. FORMULES ET COMBOS
    for f in formulas:
        fname = f["name"]
        fprice = f["price"]
        fdesc = f["description"]
        finc = " ; ".join(f["includes"])
        fcond = f.get("conditions", "Valable tous les jours.")

        add_dialogue(
            f"Parle-moi de la {fname}.",
            f"La {fname} est à {fprice:.1f} DT. Elle comprend : {finc}. {fcond}"
        )
        add_dialogue(
            f"Combien coûte la {fname} et qu'est-ce qu'elle contient ?",
            f"La {fname} coûte {fprice:.1f} DT. Elle inclut : {finc}. Note : {fcond}"
        )
        add_dialogue(
            f"9adeh soum {fname} ?",
            f"La {fname} est au prix de {fprice:.1f} DT ({fdesc}). {fcond}"
        )

    # 6. FILTRAGE PAR ALLERGÈNES & RÉGIMES ALIMENTAIRES
    diet_queries = [
        (
            "Je suis allergique au gluten, qu'est-ce que je peux manger ou boire ?",
            "Pour un régime sans gluten, nous vous recommandons : Café Express, Café Allongé, Café Arabe, Thé à la menthe, Salade Quinoa Avocat, Salade Méchouia, Escalope Grillée, Salade de Fruits frais ou nos jus de fruits frais. Évitez les viennoiseries, pizzas, pâtes et crêpes."
        ),
        (
            "Avez-vous des plats végétariens ?",
            "Absolument ! Voici plusieurs excellents choix végétariens : Sandwich Végétarien (6.5 DT), Salade Quinoa Avocat (11.5 DT), Pizza Margherita (10.0 DT), Penne Arrabbiata (10.5 DT), Omelette au fromage (7.0 DT), Pancakes et nos délicieux desserts comme le Tiramisu ou la Crêpe Nutella."
        ),
        (
            "Je suis intolérant au lactose, que me conseillez-vous ?",
            "Pour les intolérants au lactose, nous proposons : Café Express, Café Allongé, Café Arabe, Thés et infusions, Jus d'orange frais, Citronnade menthe, Salade Méchouia, Salade Quinoa Avocat, Penne Arrabbiata, Ojja Merguez et Makroudh."
        ),
        (
            "Quelles sont vos options véganes ?",
            "Nos options 100% végétaliennes (vegan) comprennent : Café Express, Café Allongé, Café Arabe, tous nos thés à la menthe et infusions, Jus d'orange pressé, Smoothie fruits rouges, Citronnade menthe, Salade Quinoa Avocat, Penne Arrabbiata et Salade de fruits frais."
        ),
        (
            "Je veux une boisson sans caféine pour le soir.",
            "Pour le soir sans caféine, nous vous conseillons nos infusions apaisantes de Verveine (2.5 DT), Camomille (2.5 DT), ou Thé Gingembre-Citron (3.0 DT), ainsi que notre Citronnade Menthe fraîche ou un Jus d'orange pressé minute."
        )
    ]
    for u, a in diet_queries:
        add_dialogue(u, a)

    # 7. FILTRAGE PAR BUDGET ET RECOMMANDATIONS
    budget_queries = [
        (
            "Je cherche quelque chose à moins de 5 dinars.",
            "À moins de 5 DT, vous avez un large choix : Café Express (2.5 DT), Capucin (2.8 DT), Café Direct (3.0 DT), Thé à la menthe (2.5 DT), Citronnade menthe (4.0 DT), Croissant pur beurre (2.0 DT), Pain au chocolat (2.5 DT), Fricassé tunisien (3.5 DT) et Makroudh au miel (3.0 DT)."
        ),
        (
            "Quel est le plat le moins cher ?",
            "Pour un repas salé économique, nous vous suggérons le Fricassé tunisien à seulement 3.5 DT, le Sandwich Végétarien à 6.5 DT, ou le Sandwich Thon à 7.0 DT. Pour les formules, la Formule Matin est à 8.5 DT et la Formule Étudiant à 12.0 DT."
        ),
        (
            "Avez-vous une formule pour deux personnes ?",
            "Oui ! Nous proposons la 'Formule Duo' à seulement 25.0 DT pour 2 personnes. Elle inclut 2 sandwichs au choix (hors Club Sandwich) et 2 boissons au choix. C'est idéal pour un repas partagé !"
        ),
        (
            "Quelle est votre spécialité tunisienne ?",
            "Nos incontournables spécialités tunisiennes sont : la Ojja Merguez mijotée à la harissa (12.5 DT), le Capucin tunisien (2.8 DT), le Thé vert aux pignons (4.5 DT), le Fricassé garni (3.5 DT), la Salade Méchouia (7.5 DT), le Tajine tunisien doré au four (8.0 DT) et le Makroudh aux dattes (3.0 DT)."
        ),
        (
            "Chnouma les sandwichs eli andkom ?",
            "3anna plusieurs sandwichs délicieux : Sandwich Poulet grillé (8.0 DT), Sandwich Thon à l'harissa (7.0 DT), Fricassé tunisien (3.5 DT), Panini Dinde Fromage (7.5 DT), Sandwich Végétarien (6.5 DT) et le copieux Club Sandwich avec frites (11.0 DT)."
        ),
        (
            "Aatini 7aja 7louwa behya",
            "Pour une touche sucrée, nous vous recommandons notre Fondant au chocolat noir chaud avec sa boule vanille (7.0 DT), le Tiramisu maison (6.0 DT), la Crêpe Nutella (6.0 DT), la Baklawa aux pistaches (4.0 DT) ou le Makroudh au miel (3.0 DT)."
        )
    ]
    for u, a in budget_queries:
        add_dialogue(u, a)

    # 8. QUESTIONS HORS-SUJET (REFUS COURTOIS)
    out_of_scope = [
        ("Est-ce que vous vendez des ordinateurs ou des smartphones ?", "Non, désolé ! Nous sommes le café-restaurant 'Coffee House Lac'. Nous servons des cafés, boissons, viennoiseries, pizzas, pâtes et plats cuisinés. Que puis-je vous proposer à déguster ?"),
        ("Peux-tu me coder un site web ?", "Je suis l'assistant dédié du Coffee House Lac ! Je ne peux pas écrire de code, mais je peux vous conseiller sur nos meilleurs cafés, plats et formules du jour !"),
        ("Quelle est la météo à Paris ?", "Je suis spécialisé uniquement sur les informations et le menu du Coffee House Lac aux Berges du Lac à Tunis. Souhaitez-vous commander une boisson chaude ou fraîche ?")
    ]
    for u, a in out_of_scope:
        add_dialogue(u, a)

    # Mélanger le dataset
    random.seed(42)
    random.shuffle(dataset)

    # Sauvegarder
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Dataset généré avec succès : {len(dataset)} dialogues complets sauvegardés dans {output_path}")
    return dataset


if __name__ == "__main__":
    menu_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "menu.json")
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_training.json")
    build_training_dataset(menu_file, out_file)
