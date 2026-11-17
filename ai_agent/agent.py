"""
=============================================================================
🤖 AGENT CONVERSATIONNEL DU SMART CAFÉ (LOCAL & OFFLINE)
=============================================================================
Cet agent répond aux questions des clients sur le menu, les prix,
les allergènes, les formules et la disponibilité des tables.

Fonctionne en 2 modes :
1. MODE MODÈLE LOCAL ENTRAÎNÉ : Si l'adaptateur LoRA issu de Google Colab
   est placé dans `ai_agent/models/cafe_agent_adapter/`, il charge le LLM local
   avec PyTorch/PEFT (100% offline, aucune API commerciale).
2. MODE MOTEUR HYBRIDE RAG : Analyse sémantique + interrogation directe de
   la base JSON du menu (`menu.json`) et du statut des tables caméras.
=============================================================================
"""

import os
import re
import json
from typing import Dict, Any, Optional

try:
    from .menu_engine import MenuEngine, normalize_text
except ImportError:
    from menu_engine import MenuEngine, normalize_text


class CafeAgent:
    def __init__(self, model_dir: Optional[str] = None):
        self.menu_engine = MenuEngine()
        self.model = None
        self.tokenizer = None
        self.mode = "rag_engine"

        # Vérifier si un modèle fine-tuné est disponible localement
        if model_dir is None:
            base = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(base, "models", "cafe_agent_adapter")
            if os.path.exists(candidate):
                model_dir = candidate

        if model_dir and os.path.exists(model_dir):
            try:
                self._load_local_llm(model_dir)
                self.mode = "local_llm"
                print(f"🧠 [Agent IA] Modèle LLM local chargé avec succès depuis : {model_dir}")
            except Exception as e:
                print(f"⚠️ [Agent IA] Impossible de charger le LLM local ({e}). Utilisation du moteur hybride local.")
                self.mode = "rag_engine"
        else:
            print("💡 [Agent IA] Modèle LoRA non présent localement. Utilisation du moteur de recherche sémantique local.")

    def _load_local_llm(self, adapter_path: str):
        """Charge le modèle Hugging Face + LoRA adapter en local."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"⏳ Chargement du tokenizer et du modèle de base ({base_model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

    def _generate_with_llm(self, user_message: str, max_new_tokens: int = 150) -> str:
        """Génération directe de réponse par le LLM entraîné avec LoRA."""
        import torch
        system_prompt = (
            "Tu es l'agent conversationnel intelligent du café-restaurant 'Coffee House Lac' à Tunis. "
            "Ton rôle est d'accueillir chaleureusement les clients, de répondre à leurs questions sur le menu, "
            "les prix en dinars tunisiens (DT), les ingrédients, les allergènes, les formules et les services du café. "
            "Tu t'exprimes avec politesse, clarté et précision en français ou dialecte tunisien."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    def _is_table_inquiry(self, norm_msg: str) -> bool:
        """Détecte avec précision si la question concerne l'occupation des tables / places / vision IoT."""
        words = norm_msg.split()
        table_words = {"table", "tables", "blasa", "blayes", "chaise", "chaises", "complet"}
        if any(w in words or w in norm_msg for w in table_words):
            return True

        # Mots relatifs aux places assises combinés à une notion de disponibilité
        place_words = {"place", "places", "asseoir"}
        if any(w in words for w in place_words):
            if any(w in words for w in ["libre", "libres", "fama", "reste", "disponible", "disponibles", "trouver"]):
                return True

        return False

    def respond(self, user_message: str, table_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Point d'entrée principal pour traiter le message du client.
        Retourne :
        {
            "reply": str,
            "intent": str,
            "items": list,
            "tables_info": dict,
            "mode": str
        }
        """
        user_clean = user_message.strip()
        norm_msg = normalize_text(user_clean)

        # Si le modèle LLM fine-tuné est disponible en local
        if self.mode == "local_llm" and self.model is not None:
            # Gestion des tables avec la caméra
            if self._is_table_inquiry(norm_msg):
                return self._handle_table_inquiry(user_clean, table_state)
            try:
                llm_reply = self._generate_with_llm(user_clean)
                return {
                    "reply": llm_reply,
                    "intent": "llm_generated",
                    "items": [],
                    "tables_info": table_state,
                    "mode": "local_llm"
                }
            except Exception as e:
                print(f"⚠️ Erreur inférence LLM ({e}), bascule sur moteur hybride.")

        # 1. Gestion des tables et vision IoT
        if self._is_table_inquiry(norm_msg):
            return self._handle_table_inquiry(user_clean, table_state)

        # 2. Gestion du Wi-Fi
        if any(w in norm_msg for w in ["wifi", "wi fi", "internet", "code", "mot de passe"]):
            wifi_note = self.menu_engine.cafe.get("wifi", {}).get("note", "Wi-Fi gratuit, le code est donné à la caisse.")
            return {
                "reply": f"📶 Le Wi-Fi est entièrement gratuit pour nos clients ! {wifi_note}",
                "intent": "cafe_info_wifi",
                "items": [],
                "mode": self.mode
            }

        # 3. Horaires & Contact
        if any(w in norm_msg for w in ["horaire", "horaires", "ouvert", "ferme", "fermeture", "ouverture", "wa9tech"]):
            h = self.menu_engine.cafe.get("hours_groups", [])
            txt = "🕒 Nos horaires d'ouverture sont :\n"
            for grp in h:
                txt += f"• {grp['days_fr'].capitalize()} : de {grp['open']} à {grp['close']}\n"
            return {
                "reply": txt.strip(),
                "intent": "cafe_info_hours",
                "items": [],
                "mode": self.mode
            }

        # 4. Parking & Adresse
        if any(w in norm_msg for w in ["parking", "garer", "voiture", "karhba"]):
            p_info = self.menu_engine.cafe.get("services", {}).get("parking", "Parking disponible devant le café.")
            return {
                "reply": f"🚗 Oui ! {p_info}.",
                "intent": "cafe_info_parking",
                "items": [],
                "mode": self.mode
            }

        if any(w in norm_msg for w in ["adresse", "ou vous", "win jey", "emplacement", "localisation"]):
            adr = self.menu_engine.cafe.get("address", "Les Berges du Lac 1, Tunis")
            return {
                "reply": f"📍 Notre adresse : {adr}. Vous êtes les bienvenus !",
                "intent": "cafe_info_address",
                "items": [],
                "mode": self.mode
            }

        # 5. Livraison
        if any(w in norm_msg for w in ["livraison", "livrer", "delivery", "twaslou"]):
            d = self.menu_engine.cafe.get("delivery", {})
            return {
                "reply": f"🛵 Oui, nous assurons la livraison de {d.get('hours', '11h à 22h')} dans les zones : {', '.join(d.get('zones', []))}. Frais : {d.get('fee', 3.0):.1f} DT (minimum de commande {d.get('min_order', 15.0):.1f} DT).",
                "intent": "cafe_info_delivery",
                "items": [],
                "mode": self.mode
            }

        # 6. Salutations
        if any(norm_msg.startswith(w) for w in ["bonjour", "salut", "bonsoir", "coucou", "aslema", "3aslema", "hello", "hi"]):
            return {
                "reply": "☕ Bonjour et marhba bik au Coffee House Lac ! Que puis-je vous servir aujourd'hui ? Vous pouvez me poser des questions sur notre menu, nos formules, nos prix ou nos spécialités.",
                "intent": "greeting",
                "items": [],
                "mode": self.mode
            }

        # 7. Formules
        if any(w in norm_msg for w in ["formule", "formules", "combo", "menu midi", "menu etudiant"]):
            matched_formulas = self.menu_engine.get_formulas(user_clean)
            reply_lines = ["🍽️ Voici nos formules avantageuses :"]
            for f in matched_formulas:
                reply_lines.append(f"• **{f['name']}** ({f['price']:.1f} DT) : {f['description']}")
            return {
                "reply": "\n".join(reply_lines),
                "intent": "formulas",
                "items": matched_formulas,
                "mode": self.mode
            }

        # 8. Catégories du menu (ex: "liste des jus", "quels sont les cafés", "fama sandwich", "les pizzas")
        cat_info = self.menu_engine.detect_category(user_clean)
        if cat_info:
            cat_items = self.menu_engine.search_items(category=cat_info["id"])
            if cat_items:
                lines = [f"{cat_info['icon']} **Voici notre sélection de {cat_info['name']} :**"]
                for it in cat_items:
                    lines.append(f"• **{it['name']}** ({it['price']:.1f} DT) : {it['description']}")
                return {
                    "reply": "\n".join(lines),
                    "intent": f"category_{cat_info['id']}",
                    "items": cat_items,
                    "mode": self.mode
                }

        # 9. Recherche d'un article spécifique (prix / description)
        item = self.menu_engine.find_item_by_name(user_clean)
        if item:
            all_str = f"⚠️ Allergènes : {', '.join(item['allergens'])}" if item.get("allergens") else "✅ Ne contient aucun allergène majeur."
            reply = f"☕ **{item['name']}** : {item['price']:.1f} DT\n{item['description']}\n{all_str}"
            if not item.get("available", True):
                reply += f"\n*(Note: {item.get('availability_note', 'Rupture temporaire')})*"
            return {
                "reply": reply,
                "intent": "item_detail",
                "items": [item],
                "mode": self.mode
            }

        # 9. Allergènes et régimes alimentaires
        if "gluten" in norm_msg:
            safe = self.menu_engine.search_items(tag="sans_gluten")
            sample = [f"{x['name']} ({x['price']:.1f} DT)" for x in safe[:5]]
            return {
                "reply": f"🌾 Pour un régime **sans gluten**, voici quelques choix recommandés :\n• " + "\n• ".join(sample),
                "intent": "diet_gluten_free",
                "items": safe[:5],
                "mode": self.mode
            }

        if "vegetarien" in norm_msg or "vege" in norm_msg or "vegan" in norm_msg:
            veg = self.menu_engine.search_items(tag="vegetarien")
            sample = [f"{x['name']} ({x['price']:.1f} DT)" for x in veg[:5]]
            return {
                "reply": f"🥗 Pour les **végétariens**, nous vous conseillons :\n• " + "\n• ".join(sample),
                "intent": "diet_vegetarian",
                "items": veg[:5],
                "mode": self.mode
            }

        if "lactose" in norm_msg or "lait" in norm_msg:
            safe = self.menu_engine.search_items(tag="sans_lactose")
            sample = [f"{x['name']} ({x['price']:.1f} DT)" for x in safe[:5]]
            return {
                "reply": f"🥛 Pour les intolérants au **lactose**, nous vous proposons :\n• " + "\n• ".join(sample),
                "intent": "diet_lactose_free",
                "items": safe[:5],
                "mode": self.mode
            }

        # 10. Budget / Moins de X dinars
        match_price = re.search(r"(\d+)\s*(dt|dinar|dinars)?", norm_msg)
        if match_price and ("moins" in norm_msg or "budget" in norm_msg or "max" in norm_msg):
            max_p = float(match_price.group(1))
            budget_items = self.menu_engine.search_items(max_price=max_p)
            sample = [f"{x['name']} ({x['price']:.1f} DT)" for x in budget_items[:6]]
            return {
                "reply": f"💰 À moins de {max_p:.0f} DT, voici d'excellentes suggestions :\n• " + "\n• ".join(sample),
                "intent": "budget_filter",
                "items": budget_items[:6],
                "mode": self.mode
            }

        # 11. Recherche générique par mot-clé
        results = self.menu_engine.search_items(query=user_clean)
        if results:
            lines = [f"🔍 Voici ce que nous avons trouvé pour '{user_clean}' :"]
            for r in results[:4]:
                lines.append(f"• **{r['name']}** ({r['price']:.1f} DT) : {r['description']}")
            return {
                "reply": "\n".join(lines),
                "intent": "menu_search",
                "items": results[:4],
                "mode": self.mode
            }

        # 12. Fallback
        return {
            "reply": "Je n'ai pas tout à fait compris votre demande. Vous pouvez me demander par exemple :\n"
                     "• 'Combien coûte le capucin ?'\n"
                     "• 'Avez-vous des plats sans gluten ?'\n"
                     "• 'Quelles sont vos formules du midi ?'\n"
                     "• 'Y a-t-il des tables libres en ce moment ?'\n"
                     "• 'C'est quoi le code wifi ?'",
            "intent": "fallback",
            "items": [],
            "mode": self.mode
        }

    def _handle_table_inquiry(self, query: str, table_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Gère les questions relatives à l'occupation des tables en direct (Vision IoT)."""
        if table_state and "free_count" in table_state:
            free = table_state.get("free_count", 0)
            occupied = table_state.get("occupied_count", 0)
            total = table_state.get("total_tables", free + occupied)
            if free > 0:
                reply = f"🪑 Oui ! Selon notre caméra de surveillance intelligente en direct, il y a **{free} table(s) libre(s)** sur {total} ({occupied} occupée(s)). Vous pouvez vous installer immédiatement !"
            else:
                reply = f"⌛ Actuellement, toutes nos {total} tables sont occupées d'après le système de détection vidéo. Une table devrait se libérer sous peu !"
            return {
                "reply": reply,
                "intent": "table_status_live",
                "items": [],
                "tables_info": table_state,
                "mode": self.mode
            }
        else:
            return {
                "reply": "🪑 D'après notre système de surveillance en direct, des tables sont actuellement libres au café (en salle climatisée et sur la terrasse). Marhba bik !",
                "intent": "table_status_default",
                "items": [],
                "mode": self.mode
            }
