"""
=============================================================================
☕ CONTROLLER - VISION ROUTES (Sources Switch & Video Stream)
=============================================================================
Endpoints pour :
- Le flux MJPEG (/video_feed)
- L'état des tables (/api/status)
- La recalibration (/api/recalibrate)
- La sélection dynamique de source vidéo (/api/vision/sources, /api/vision/source)
=============================================================================
"""

from flask import Blueprint, Response, jsonify, request
from ..services.vision_service import VisionService

vision_bp = Blueprint("vision", __name__)
_vision_service: VisionService = None


def init_vision_routes(vision_service: VisionService):
    global _vision_service
    _vision_service = vision_service


@vision_bp.route("/video_feed")
def video_feed():
    """Flux vidéo MJPEG en direct pour Web / Mobile / Angular."""
    if not _vision_service:
        return "Vision service not initialized", 503
    return Response(
        _vision_service.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@vision_bp.route("/api/status")
def api_status():
    """Retourne l'état complet en temps réel des tables (total, libres, occupées, timers)."""
    if not _vision_service:
        return jsonify({"error": "Vision non initialisée"}), 503
    state = _vision_service.get_latest_state()
    res = jsonify(state)
    res.headers.add("Access-Control-Allow-Origin", "*")
    return res


@vision_bp.route("/api/vision/sources", methods=["GET"])
def api_get_sources():
    """Retourne la liste des sources disponibles (Webcam + Vidéos) et la source active."""
    if not _vision_service:
        return jsonify({"error": "Vision non initialisée"}), 503
    data = _vision_service.get_available_sources()
    res = jsonify(data)
    res.headers.add("Access-Control-Allow-Origin", "*")
    return res


@vision_bp.route("/api/vision/source", methods=["POST"])
def api_set_source():
    """Bascule dynamiquement la source vidéo (ex: { "source": "0" } ou { "source": "cafe.mp4" })."""
    if not _vision_service:
        return jsonify({"error": "Vision non initialisée"}), 503

    payload = request.get_json(silent=True) or {}
    new_src = payload.get("source", "").strip()

    if not new_src:
        return jsonify({"error": "Paramètre 'source' manquant."}), 400

    success = _vision_service.switch_source(new_src)
    res = jsonify({
        "success": success,
        "message": f"Source basculée vers '{new_src}'.",
        "current_source": new_src
    })
    res.headers.add("Access-Control-Allow-Origin", "*")
    return res


@vision_bp.route("/api/recalibrate", methods=["GET", "POST"])
def api_recalibrate():
    """Déclenche la recalibration des tables."""
    if not _vision_service:
        return jsonify({"success": False, "message": "Vision non initialisée"}), 503
    success = _vision_service.recalibrate()
    return jsonify({
        "success": success,
        "message": "Calibration réinitialisée avec succès." if success else "Échec de recalibration."
    })
