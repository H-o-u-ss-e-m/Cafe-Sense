"""
=============================================================================
☕ SERVICE - AGENT SERVICE
=============================================================================
Orchestre l'agent conversationnel IA, fait le pont avec le modèle de vision
(état réel des tables) et la base du menu.
=============================================================================
"""

import sys
from typing import Dict, Any, Optional
from ..core.config import Config
from ..domain.chat_message import ChatResponse

# Import de CafeAgent depuis ai_agent
sys.path.append(Config.AI_AGENT_DIR)
from agent import CafeAgent


class AgentService:
    def __init__(self):
        print("🤖 [AgentService] Initialisation du moteur d'agent IA...")
        self.agent = CafeAgent()

    def process_message(self, message: str, table_state: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """Traite le message d'un client et renvoie un ChatResponse typé."""
        raw_result = self.agent.respond(message, table_state=table_state)
        return ChatResponse(
            reply=raw_result.get("reply", ""),
            intent=raw_result.get("intent", "unknown"),
            mode=raw_result.get("mode", "rag_engine"),
            items=raw_result.get("items", []),
            tables_info=raw_result.get("tables_info", table_state)
        )
