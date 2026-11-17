"""
=============================================================================
☕ SCRIPT DE TEST INTERACTIF DE L'AGENT IA SMART CAFÉ
=============================================================================
Permet de tester l'agent conversationnel directement dans le terminal :
- Mode interactif (dialogue libre avec l'agent)
- Mode batterie de tests automatiques (Français, Tunisien/Derja, Allergènes, Tables)
- Fonctionne immédiatement sans clé API, 100% en local !
=============================================================================
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai_agent.agent import CafeAgent


def run_automated_tests(agent: CafeAgent):
    print("\n" + "="*70)
    print("🧪 LANCEMENT DE LA BATTERIE DE TESTS AUTOMATIQUES DU SMART CAFÉ")
    print("="*70)

    test_queries = [
        ("Salutation", "Bonjour, je voudrais des renseignements"),
        ("Prix article (FR)", "Combien coûte le capucin ?"),
        ("Prix article (Tunisien)", "9adeh soum l'express ?"),
        ("Ingrédients & description", "Qu'est-ce qu'il y a dans l'ojja merguez ?"),
        ("Régime sans gluten", "Je suis allergique au gluten, que puis-je manger ?"),
        ("Plats végétariens", "Avez-vous des options végétariennes ?"),
        ("Formule pour deux", "Avez-vous une formule pour 2 personnes ?"),
        ("Filtre budget", "Je cherche quelque chose à moins de 4 dinars"),
        ("Informations Wi-Fi", "Quel est le mot de passe wifi ?"),
        ("Horaires du café", "Quels sont vos horaires du week-end ?"),
        ("Livraison", "Faites-vous la livraison à La Marsa ?"),
        ("Disponibilité des tables (Vision)", "Est-ce qu'il y a des tables libres en ce moment ?"),
    ]

    mock_table_state = {
        "total_tables": 4,
        "free_count": 2,
        "occupied_count": 2,
        "tables": [
            {"id": 1, "status": "LIBRE", "timer": "00:00:00"},
            {"id": 2, "status": "OCCUPEE", "timer": "00:15:30"},
            {"id": 3, "status": "LIBRE", "timer": "00:00:00"},
            {"id": 4, "status": "OCCUPEE", "timer": "00:42:10"},
        ]
    }

    for category, query in test_queries:
        print(f"\n📌 [{category.upper()}]")
        print(f"👤 CLIENT : {query}")
        resp = agent.respond(query, table_state=mock_table_state)
        print(f"☕ AGENT  : {resp['reply']}")
        print(f"⚙️ [Intent: {resp['intent']} | Mode: {resp['mode']}]")
        print("-" * 50)

    print("\n✅ Batterie de tests terminée avec succès !")


def interactive_chat(agent: CafeAgent):
    print("\n" + "="*70)
    print("💬 SESSION INTERACTIVE AVEC L'AGENT DU CAFÉ (Tapez 'quit' pour quitter)")
    print("="*70)
    print("Exemples de questions à poser :")
    print(" • 'Combien coûte le sandwich poulet ?'")
    print(" • '9adeh soum l'express ?'")
    print(" • 'Je suis allergique au gluten, qu'est-ce que vous me conseillez ?'")
    print(" • 'Avez-vous une formule pour étudiant ?'")
    print(" • 'C'est quoi le code wifi ?'")
    print(" • 'Est-ce qu'il y a des tables libres ?'")
    print("="*70 + "\n")

    mock_state = {"total_tables": 4, "free_count": 3, "occupied_count": 1}

    while True:
        try:
            user_input = input("👤 Vous > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Au revoir et à bientôt au Coffee House Lac !")
                break

            response = agent.respond(user_input, table_state=mock_state)
            print(f"\n☕ Agent > {response['reply']}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session fermée.")
            break


def main():
    agent = CafeAgent()
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        run_automated_tests(agent)
    else:
        run_automated_tests(agent)
        interactive_chat(agent)


if __name__ == "__main__":
    main()
