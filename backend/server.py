"""
=============================================================================
☕ SMART CAFE SERVER - CLEAN ARCHITECTURE ENTRYPOINT
=============================================================================
Fournit :
  1. Le flux vidéo MJPEG avec détections, couleurs et chronos (/video_feed)
  2. L'API REST JSON pour Angular (/api/status, /api/chat, /api/menu, /api/recalibrate)
  3. L'Agent IA local branché sur la détection temps réel des tables
=============================================================================
"""

import argparse
import sys
import os

# Configuration du PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.factory import create_app
from backend.app.core.config import Config


def main():
    parser = argparse.ArgumentParser(description="Serveur Backend Smart Café (Clean Architecture)")
    parser.add_argument("--source", type=str, default=Config.DEFAULT_VIDEO if os.path.exists(Config.DEFAULT_VIDEO) else "0",
                        help="Index caméra ou chemin vidéo (ex: '0', 'CAM1.mp4', 'CAM2.mp4', 'CAM3.mp4')")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="Nom ou chemin vers le modèle YOLO (.pt ou .onnx)")
    parser.add_argument("--port", type=int, default=Config.PORT,
                        help="Port du serveur HTTP (défaut : 5000)")
    parser.add_argument("--table-conf", type=float, default=Config.TABLE_CONF,
                        help="Confiance minimale pour détection des tables")
    parser.add_argument("--person-conf", type=float, default=Config.PERSON_CONF,
                        help="Confiance minimale pour détection des personnes")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confiance globale optionnelle pour écraser les deux seuils")
    parser.add_argument("--imgsz", type=int, default=Config.IMGSZ,
                        help="Taille image inférence (défaut: 640)")
    parser.add_argument("--skip", type=int, default=Config.SKIP_FRAMES,
                        help="Traiter 1 image toutes les N (1=toutes)")
    args = parser.parse_args()

    t_conf = args.conf if args.conf is not None else args.table_conf
    p_conf = args.conf if args.conf is not None else args.person_conf

    app = create_app(
        source=args.source,
        model_path=args.model,
        table_conf=t_conf,
        person_conf=p_conf,
        imgsz=args.imgsz,
        skip_frames=args.skip
    )

    print(f"\n" + "="*70)
    print(f"🚀 Serveur Flask (Clean Architecture) démarré !")
    print(f"📡 API Status Tables : http://localhost:{args.port}/api/status")
    print(f"🤖 API Chat Agent IA : http://localhost:{args.port}/api/chat (POST)")
    print(f"🍽️ API Catalogue Menu: http://localhost:{args.port}/api/menu")
    print(f"📺 Flux Vidéo MJPEG  : http://localhost:{args.port}/video_feed")
    print(f"🌐 Frontend Angular  : http://localhost:4200 (autorisé via CORS)")
    print("="*70 + "\n")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
