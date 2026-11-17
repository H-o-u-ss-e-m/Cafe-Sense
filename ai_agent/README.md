# 🤖 Module Agent IA - Smart Café

Ce module implémente l'agent conversationnel intelligent du café :
- **100% autonome et local** (sans aucune API externe payante).
- Entraîné sous **Google Colab** avec **QLoRA** sur le modèle multilingue **Qwen2.5-1.5B**.
- Connecté à la base de données JSON du café (`data/menu.json`) et au système de détection vidéo des tables (`ai_vision`).

## Structure des Fichiers

- `colab/Entrainement_Agent_Cafe_Colab.ipynb` : Notebook prêt à l'emploi pour Google Colab (avec GPU T4).
- `data/menu.json` : Base de données complète du café (68 articles, 6 formules, prix en DT, allergènes, horaires, wifi).
- `data/dataset_training.json` : Dataset d'instructions généré (391 dialogues en français, dialecte tunisien et anglais).
- `generate_dataset.py` : Script de génération et d'enrichissement du jeu de données.
- `menu_engine.py` : Moteur de recherche multicritère (fuzzy matching, filtres allergènes, budget, catégories).
- `agent.py` : Moteur de l'agent IA (supporte le modèle local ou le moteur sémantique hybride).
- `test_agent.py` : Script de test interactif dans la console.

## Lancer le Test Local

```bash
python ai_agent/test_agent.py
```
