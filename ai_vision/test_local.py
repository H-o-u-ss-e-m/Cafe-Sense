"""
=============================================================================
💻 SCRIPT DE TEST LOCAL - CAFE MONITOR V2 (PC / WEBCAM / VIDÉO)
=============================================================================
Utilisation :
  - Sur webcam :            python test_local.py --source 0
  - Sur fichier vidéo :     python test_local.py --source resultat.mp4
  - Avec modèle spécifique: python test_local.py --model best.pt --source 0
Touches :
  - 'q' : Quitter
  - 'r' : Relancer la calibration automatique des tables
  - 's' : Prendre une capture d'écran
=============================================================================
"""

import argparse
import os
import sys
import time
import cv2
from cafe_monitor_v2 import CafeMonitorV2


def main():
    parser = argparse.ArgumentParser(description="Test Local - Détection de Tables et Gestion de Café")
    parser.add_argument("--source", type=str, default="0",
                        help="Source vidéo : '0' pour webcam, ou nom/chemin vers vidéo (ex: CAM1.mp4, CAM2.mp4, CAM3.mp4)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Chemin vers le modèle YOLO (.pt ou .onnx)")
    parser.add_argument("--table-conf", type=float, default=0.35,
                        help="Seuil de confiance minimum pour les tables (défaut: 0.35)")
    parser.add_argument("--person-conf", type=float, default=0.35,
                        help="Seuil de confiance minimum pour les personnes (défaut: 0.35)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Seuil global optionnel pour écraser les deux seuils")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Résolution d'inférence (ex: 640 ou 480)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Facteur d'accélération du temps simulé pour tester les timers")
    args = parser.parse_args()

    t_conf = args.conf if args.conf is not None else args.table_conf
    p_conf = args.conf if args.conf is not None else args.person_conf

    # Détection de la source
    src = int(args.source) if args.source.isdigit() else args.source
    if isinstance(src, str) and not os.path.exists(src):
        # Chercher également dans le dossier test/
        test_cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test", os.path.basename(src))
        if os.path.exists(test_cand):
            src = test_cand
        else:
            print(f"❌ Erreur : Le fichier vidéo '{src}' n'existe pas.")
            sys.exit(1)

    model_path = args.model
    if not os.path.exists(model_path):
        candidate = os.path.join(os.path.dirname(__file__), "models", args.model)
        if os.path.exists(candidate):
            model_path = candidate
        else:
            default_candidate = os.path.join(os.path.dirname(__file__), "models", "best.pt")
            if os.path.exists(default_candidate):
                model_path = default_candidate

    print(f"🚀 Initialisation du système CafeMonitorV2 avec '{model_path}'...")
    print(f"   Seuil tables: {t_conf:.2f} | Seuil personnes: {p_conf:.2f}")
    try:
        monitor = CafeMonitorV2(
            model_path=model_path,
            table_conf=t_conf,
            person_conf=p_conf,
            imgsz=args.imgsz
        )
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        print("💡 Astuce : Assurez-vous d'avoir installé ultralytics : pip install ultralytics opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la source vidéo : {src}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"✅ Flux vidéo ouvert. FPS source : {fps_in:.1f}")
    print("📋 Raccourcis clavier : [Q] Quitter | [R] Recalibrer | [S] Capture")

    prev_time = time.time()
    sim_time_start = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Si c'est une vidéo, on peut boucler ou terminer
            if isinstance(src, str):
                print("🔄 Fin de la vidéo. Relecture...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        now = time.time()
        fps_calc = 1.0 / max(1e-4, now - prev_time)
        prev_time = now

        # Calcul du temps (en temps réel ou accéléré)
        if args.speed != 1.0:
            current_sim_time = (now - sim_time_start) * args.speed
        else:
            current_sim_time = now

        # Traitement par le moniteur
        annotated, state = monitor.process(frame, current_time=current_sim_time)

        # Affichage du FPS local
        cv2.putText(
            annotated, f"FPS: {fps_calc:.1f}", (annotated.shape[1] - 110, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )

        cv2.imshow("Smart Cafe Monitor - Methode 2 (Temps Reel)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Arrêt demandé par l'utilisateur.")
            break
        elif key == ord('r'):
            print("🔄 Réinitialisation de la calibration des tables...")
            monitor.reset_calibration()
        elif key == ord('s'):
            out_filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(out_filename, annotated)
            print(f"📸 Capture enregistrée : {out_filename}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Programme terminé.")


if __name__ == "__main__":
    main()
