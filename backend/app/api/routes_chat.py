"""
=============================================================================
☕ CONTROLLER - CHAT ROUTES
=============================================================================
Endpoint /api/chat pour dialoguer avec l'agent IA conversationnel.
=============================================================================
"""

from flask import Blueprint, request, jsonify
from ..services.agent_service import AgentService
from ..services.vision_service import VisionService

chat_bp = Blueprint("chat", __name__)
_agent_service: AgentService = None
_vision_service: VisionService = None


def init_chat_routes(agent_service: AgentService, vision_service: VisionService):
    global _agent_service, _vision_service
    _agent_service = agent_service
    _vision_service = vision_service


@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Reçoit : { "message": "Combien coûte le capucin ?" }
    Retourne la réponse de l'agent IA + l'état des tables en direct.
    """
    if not _agent_service:
        return jsonify({"error": "Service d'agent IA non disponible"}), 503

    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Le champ 'message' est obligatoire."}), 400

    table_state = _vision_service.get_latest_state() if _vision_service else None
    response_obj = _agent_service.process_message(message, table_state=table_state)

    res = jsonify(response_obj.to_dict())
    res.headers.add("Access-Control-Allow-Origin", "*")
    return res
