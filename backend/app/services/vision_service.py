"""
=============================================================================
☕ SERVICE - VISION SERVICE (Clean Architecture & Dynamic Source Switch)
=============================================================================
Gère le flux vidéo, la détection YOLO temps réel avec CafeMonitorV2,
l'encodage MJPEG et le BASCULEMENT DYNAMIQUE entre la Webcam et les Vidéos (cafe.mp4, resultat.mp4).
=============================================================================
"""

import os
import sys
import time
import threading
import cv2
from typing import Dict, Any, Generator, Optional, List
from ..core.config import Config

# Import dynamique de CafeMonitorV2
sys.path.append(Config.AI_VISION_DIR)
from cafe_monitor_v2 import CafeMonitorV2


class VisionService:
    def __init__(
        self,
        source: Optional[str] = None,
        model_path: Optional[str] = None,
        table_conf: float = Config.TABLE_CONF,
        person_conf: float = Config.PERSON_CONF,
        imgsz: int = Config.IMGSZ,
        skip_frames: int = Config.SKIP_FRAMES
    ):
        # Par défaut, essayer la webcam "0" ou fallback sur video si pas dispo
        self.source = source or "0"
        self.model_path = model_path or Config.DEFAULT_MODEL
        self.table_conf = table_conf
        self.person_conf = person_conf
        self.imgsz = imgsz
        self.skip_frames = skip_frames

        self.monitor: Optional[CafeMonitorV2] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pending_source: Optional[str] = None

        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._current_jpeg: Optional[bytes] = None
        self._latest_state: Dict[str, Any] = {
            "total_tables": 0,
            "free_count": 0,
            "occupied_count": 0,
            "leaving_count": 0,
            "tables": []
        }

    def _resolve_source_path(self, src: str) -> Any:
        """Résout un chemin de vidéo ou un index de caméra numérique."""
        if str(src).isdigit():
            return int(src)
        
        base = os.path.basename(src)
        base_no_ext, ext = os.path.splitext(base)

        # Variations de casse (CAM1.mp4, cam1.mp4, etc.)
        name_variants = [
            base,
            f"{base_no_ext.lower()}{ext.lower()}",
            f"{base_no_ext.upper()}{ext.lower()}",
            f"{base_no_ext.capitalize()}{ext.lower()}"
        ]

        candidates = [src]
        for v in name_variants:
            candidates.extend([
                os.path.join(Config.TEST_DIR, v),
                os.path.join(Config.PROJECT_ROOT, "test", v),
                os.path.join(Config.PROJECT_ROOT, v),
                os.path.join(Config.AI_VISION_DIR, v)
            ])

        for c in candidates:
            if os.path.exists(c):
                return c
        return src

    def get_available_sources(self) -> Dict[str, Any]:
        """Retourne la liste des sources vidéo sélectionnables (Webcam PC + CAM1, CAM2, CAM3)."""
        sources = [
            {
                "id": "0",
                "name": "📹 Caméra PC (Webcam Direct)",
                "type": "webcam",
                "available": True
            }
        ]

        cam_definitions = [
            ("CAM1.mp4", "🎥 Caméra 1 - Salle Principale"),
            ("CAM2.mp4", "🎥 Caméra 2 - Terrasse"),
            ("CAM3.mp4", "🎥 Caméra 3 - Comptoir & Entrée")
        ]

        for cam_file, cam_label in cam_definitions:
            resolved = self._resolve_source_path(cam_file)
            if isinstance(resolved, str) and os.path.exists(resolved):
                sources.append({
                    "id": cam_file,
                    "name": cam_label,
                    "type": "video",
                    "available": True
                })

        return {
            "current_source": str(self.source),
            "sources": sources
        }

        return {
            "current_source": str(self.source),
            "sources": sources
        }

    def switch_source(self, new_source: str) -> bool:
        """Demande le basculement dynamique vers une nouvelle source vidéo."""
        print(f"🔄 [VisionService] Demande de basculement vers : {new_source}")
        with self._lock:
            self._pending_source = str(new_source)
        return True

    def start(self):
        """Démarre le thread de traitement vidéo en arrière-plan."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Arrête proprement le thread."""
        self._running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def _open_capture(self, raw_source: str) -> cv2.VideoCapture:
        resolved = self._resolve_source_path(raw_source)
        if isinstance(resolved, int):
            # Sur Windows, essayer cv2.CAP_DSHOW en premier pour un démarrage rapide sans freeze
            if sys.platform.startswith("win"):
                try:
                    cap = cv2.VideoCapture(resolved, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap
                    cap.release()
                except Exception:
                    pass
            cap = cv2.VideoCapture(resolved)
        else:
            cap = cv2.VideoCapture(resolved)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def _processing_loop(self):
        # 1. Résolution et ouverture initiale
        raw_source = self.source
        self.cap = self._open_capture(raw_source)

        # Si la webcam ne s'ouvre pas, fallback automatique sur une vidéo existante
        if not self.cap.isOpened() and str(raw_source).isdigit():
            print("⚠️ [VisionService] Webcam 0 inaccessible, bascule automatique sur CAM1.mp4...")
            fallback_video = self._resolve_source_path("CAM1.mp4")
            if not os.path.exists(str(fallback_video)):
                fallback_video = self._resolve_source_path("CAM2.mp4")
            raw_source = str(fallback_video)
            self.source = raw_source
            self.cap = self._open_capture(raw_source)

        if not self.cap.isOpened():
            print(f"❌ [VisionService] Impossible d'ouvrir la source vidéo : {raw_source}")
            self._running = False
            return

        # 2. Modèle YOLO
        model_file = self.model_path
        candidates = [
            model_file,
            os.path.join(Config.AI_VISION_DIR, "models", "best.pt"),
            os.path.join(Config.AI_VISION_DIR, "models", "best.onnx"),
            os.path.join(Config.AI_VISION_DIR, "models", "yolov8n.pt"),
            os.path.join(Config.BASE_DIR, "best.pt")
        ]
        for c in candidates:
            if c and os.path.exists(c):
                model_file = c
                break

        print(f"🧠 [VisionService] Modèle actif : {model_file} | Source : {self.source}")
        self.monitor = CafeMonitorV2(
            model_path=model_file,
            table_conf=self.table_conf,
            person_conf=self.person_conf,
            imgsz=self.imgsz
        )

        frame_count = 0
        while self._running:
            # Vérifier si un changement dynamique de source a été demandé
            if self._pending_source is not None:
                with self._lock:
                    target_src = self._pending_source
                    self._pending_source = None

                print(f"🔀 [VisionService] Basculement en cours vers : {target_src}")
                if self.cap and self.cap.isOpened():
                    self.cap.release()

                new_cap = self._open_capture(target_src)
                if new_cap.isOpened():
                    self.cap = new_cap
                    self.source = target_src
                    if self.monitor:
                        self.monitor.reset_calibration()
                    print(f"✅ [VisionService] Source changée avec succès vers : {target_src}")
                else:
                    print(f"❌ [VisionService] Échec d'ouverture de la source : {target_src}. Fallback vidéo automatique.")
                    fallback_video = self._resolve_source_path("CAM1.mp4")
                    if not os.path.exists(str(fallback_video)):
                        fallback_video = self._resolve_source_path("CAM2.mp4")
                    self.cap = self._open_capture(str(fallback_video))
                    self.source = str(fallback_video)

            ret, frame = self.cap.read()
            if not ret:
                resolved = self._resolve_source_path(self.source)
                if isinstance(resolved, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame_count % self.skip_frames != 0:
                continue

            annotated, state = self.monitor.process(frame)
            ret_enc, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

            if ret_enc:
                with self._lock:
                    self._current_jpeg = buffer.tobytes()
                    self._latest_state = state
                self._frame_event.set()

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """Générateur MJPEG sans latence."""
        while self._running:
            self._frame_event.wait(timeout=0.1)
            self._frame_event.clear()

            with self._lock:
                data = self._current_jpeg

            if data is None:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")

    def get_latest_state(self) -> Dict[str, Any]:
        """Retourne l'état actuel des tables."""
        with self._lock:
            return dict(self._latest_state)

    def recalibrate(self) -> bool:
        """Déclenche la recalibration des tables."""
        if self.monitor:
            self.monitor.reset_calibration()
            return True
        return False
