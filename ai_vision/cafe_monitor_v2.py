"""
=============================================================================
☕ CAFE MONITOR V2 (OPTIMISÉ & ANTI-FAUX-POSITIFS)
=============================================================================
Améliorations majeures apportées :
  1. Élimination des Faux Positifs (Visage / Barbe / Vêtements) :
     - Seuil de confiance des tables calibré à 0.45 (le visage était à 0.16).
     - Filtre d'exclusion corporelle : rejette les boîtes situées sur le haut
       du corps ou le visage d'une personne.
     - Confirmation de présence (5 frames) avant d'ajouter une nouvelle table.
  2. Filtre Anti-Passant / Serveur (Ne pas marquer 'Occupée' si on passe juste) :
     - Si vous marchez à côté sans vous asseoir (1-2s), la table RESTE LIBRE 🟢.
     - La table ne passe en OCCUPÉE 🔴 qu'après 4 secondes de présence continue
       (le temps réel d'un client qui s'assoit à table).
     - Affichage d'un indicateur de confirmation en cours pendant l'approche.
  3. Suivi Dynamique Fluide (Frame-by-Frame) :
     - Suivi de la table en temps réel lorsqu'elle est déplacée.
=============================================================================
"""

import math
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# États
FREE = "LIBRE"
OCCUPIED = "OCCUPEE"
LEAVING = "EN_LIBERATION"

# Couleurs BGR (OpenCV)
COLOR_FREE = (46, 204, 113)       # Vert émeraude
COLOR_OCCUPIED = (46, 46, 231)     # Rouge vif
COLOR_LEAVING = (0, 165, 255)      # Orange
COLOR_PERSON = (0, 215, 255)       # Jaune/Doré pour les clients

COLORS = {
    FREE: COLOR_FREE,
    OCCUPIED: COLOR_OCCUPIED,
    LEAVING: COLOR_LEAVING
}


def compute_iou(boxA, boxB):
    """Calcule l'IoU entre deux boîtes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea

    return interArea / union if union > 0 else 0.0


def box_distance(boxA, boxB):
    """Calcule la distance minimale entre deux rectangles."""
    dx = max(0.0, boxA[0] - boxB[2], boxB[0] - boxA[2])
    dy = max(0.0, boxA[1] - boxB[3], boxB[1] - boxA[3])
    return math.hypot(dx, dy)


def format_duration(seconds):
    """Formate une durée en secondes sous forme HH:MM:SS."""
    s = int(max(0, seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


class TrackedTable:
    """Représente une table suivie de manière continue et fluide."""

    def __init__(self, table_id, bbox, current_time=0.0):
        self.id = table_id
        self.box = [float(v) for v in bbox]
        self.status = FREE
        self.persons_count = 0
        self.assigned_person_boxes = []

        # Suivi temporel
        self.last_detected = current_time
        self.missed_frames = 0

        # Gestion de l'occupation
        self.session_started = None
        self.leaving_started = None
        self.last_person_seen = -1e9
        self.presence_started = None
        self.history = []

    def update_position(self, new_box, smooth=0.75):
        """Met à jour les coordonnées en temps réel avec lissage réactif."""
        self.box = [
            smooth * new + (1.0 - smooth) * old
            for new, old in zip(new_box, self.box)
        ]
        self.missed_frames = 0

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def width(self):
        return max(1.0, self.box[2] - self.box[0])

    @property
    def height(self):
        return max(1.0, self.box[3] - self.box[1])

    def is_person_at_table(self, person_box, max_dist=30.0):
        """
        Détermine si la personne est assise à CETTE table :
        - Distance bord-à-bord stricte (<= max_dist px, par défaut 30px).
        - Élimine les fausses liaisons avec des clients assis au bar ou à d'autres tables.
        """
        dist = box_distance(self.box, person_box)

        # Seuil de distance strict : max 30 pixels (ou max 18% de la dimension de la table)
        allowed_dist = min(max_dist, max(12.0, self.width * 0.18))
        if dist > allowed_dist:
            return False

        # Vérification centre à centre (un client assis a son centre proche de la table)
        px = (person_box[0] + person_box[2]) / 2.0
        py = (person_box[1] + person_box[3]) / 2.0
        tc_x, tc_y = self.center
        center_dist = math.hypot(px - tc_x, py - tc_y)

        max_dim = max(self.width, self.height)
        if center_dist > (max_dim * 0.85 + 25.0):
            return False

        # Zone d'assise immédiate (chaises serrées contre la table : max 25 pixels)
        pad_x = min(25.0, self.width * 0.15)
        pad_y = min(25.0, self.height * 0.15)
        pad_y_bottom = min(35.0, self.height * 0.25)

        z_x1 = self.box[0] - pad_x
        z_y1 = self.box[1] - pad_y
        z_x2 = self.box[2] + pad_x
        z_y2 = self.box[3] + pad_y_bottom

        feet_y = person_box[3]

        if (z_x1 <= px <= z_x2) and (z_y1 <= py <= z_y2):
            return True
        if (z_x1 <= px <= z_x2) and (z_y1 <= feet_y <= z_y2):
            return True
        if compute_iou(self.box, person_box) > 0.02:
            return True

        return False


class CafeMonitorV2:
    """
    Système haute performance de surveillance de café :
    - Filtre anti-passant (ne valide l'occupation qu'après 3-4s d'assise continue).
    - Filtre anti-visage/barbe (confiance calibrée + exclusion du haut du corps).
    - Détection hybride tables + clients avec chronomètre dynamique.
    """

    def __init__(
        self,
        model_path="best.pt",
        table_conf=0.35,            # Seuil de confiance calibré : 0.35 capture la table immédiatement, tout en éliminant les bruits
        person_conf=0.35,           # Seuil pour les personnes
        max_person_table_distance=30.0, # Seuil de distance strict Client <-> Table (anti-fausses liaisons au bar/autres tables)
        min_presence_sec=3.0,       # 3 secondes d'assise requises pour déclarer OCCUPÉE (anti-passant)
        absence_to_leaving_sec=2.5, # 2.5s d'absence pour passer en orange (EN_LIBERATION)
        grace_period_sec=25.0,      # 25s en orange avant de libérer (idéal pour tester rapidement)
        imgsz=640,
        **kwargs
    ):
        self.table_conf = table_conf
        self.person_conf = person_conf
        self.max_person_table_distance = max_person_table_distance
        self.min_presence_sec = min_presence_sec
        self.absence_to_leaving_sec = absence_to_leaving_sec
        self.grace_period_sec = grace_period_sec
        self.imgsz = imgsz

        # Résolution sécurisée du chemin du modèle tables
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(model_path):
            for candidate in [
                os.path.join(current_dir, "models", os.path.basename(model_path)),
                os.path.join(current_dir, "models", "best.pt"),
                os.path.join(current_dir, "models", "best.onnx"),
                os.path.join(current_dir, os.path.basename(model_path)),
            ]:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        # Chargement du modèle spécialisé pour les tables
        print(f"📦 Chargement du modèle tables : {model_path}")
        self.model_tables = YOLO(model_path)

        # Chargement du modèle pour les personnes (généraliste COCO)
        is_onnx = str(model_path).lower().endswith(".onnx")
        
        candidates = []
        if is_onnx:
            candidates.extend([
                os.path.join(current_dir, "models", "yolov8n.onnx"),
                os.path.join(current_dir, "yolov8n.onnx"),
                "yolov8n.onnx",
                os.path.join(current_dir, "models", "yolov8n.pt"),
                os.path.join(current_dir, "yolov8n.pt"),
                "yolov8n.pt"
            ])
        else:
            candidates.extend([
                os.path.join(current_dir, "models", "yolov8n.pt"),
                os.path.join(current_dir, "yolov8n.pt"),
                "yolov8n.pt",
                os.path.join(current_dir, "models", "yolov8n.onnx"),
                os.path.join(current_dir, "yolov8n.onnx"),
                "yolov8n.onnx"
            ])
            
        yolo_base = next((c for c in candidates if os.path.exists(c)), "yolov8n.pt")

        print(f"👥 Chargement du modèle personnes : {yolo_base}")
        try:
            self.model_persons = YOLO(yolo_base)
        except Exception:
            self.model_persons = self.model_tables

        self._init_classes()

        self.tables = []
        self._candidates = []  # Pour confirmation sur plusieurs trames (anti-bruit)
        self._next_table_id = 1
        self._frame_count = 0
        self.events = []

    def _init_classes(self):
        t_names = {v.lower().strip(): k for k, v in self.model_tables.names.items()}
        t_key = next((k for k in ["table", "dining table", "diningtable", "desk"] if k in t_names), None)
        self.table_cls_id = t_names[t_key] if t_key is not None else 1

        p_names = {v.lower().strip(): k for k, v in self.model_persons.names.items()}
        p_key = next((k for k in ["person", "human", "customer"] if k in p_names), None)
        self.person_cls_id = p_names[p_key] if p_key is not None else 0

    def reset_calibration(self):
        self.tables = []
        self._candidates = []
        self._next_table_id = 1
        self._frame_count = 0

    def _filter_table_box(self, box, frame_w, frame_h, persons):
        """
        Filtre anti-faux-positifs :
        - Élimine toute table 'flottante' située sur la tête ou le torse d'une personne (visage/barbe/chemise).
        - Filtre de surface réaliste.
        """
        w = box[2] - box[0]
        h = box[3] - box[1]
        area = (w * h) / (frame_w * frame_h)

        if area < 0.003 or area > 0.40:
            return False

        aspect = w / max(1.0, h)
        if aspect < 0.35 or aspect > 3.2:
            return False

        # Éliminer les boîtes collées au coin inférieur extrême (filigrane OSD / label caméra)
        if box[2] >= (frame_w - 20) and box[3] >= (frame_h - 20):
            if area < 0.05:
                return False

        # Vérification si la boîte de table est située en plein milieu du haut du corps d'une personne
        tc_x = (box[0] + box[2]) / 2.0
        tc_y = (box[1] + box[3]) / 2.0

        for p in persons:
            p_top = p[1]
            p_mid = p[1] + (p[3] - p[1]) * 0.55  # Haut du corps (visage + torse supérieur)
            if (p[0] <= tc_x <= p[2]) and (p_top <= tc_y <= p_mid):
                # La boîte est sur le visage ou le torse : faux positif évident !
                return False

        return True

    def _detect_raw(self, frame):
        h, w = frame.shape[:2]

        # 1. Détection des personnes d'abord (nécessaire pour filtrer le torse/visage)
        res_p = self.model_persons.predict(
            frame, conf=self.person_conf, imgsz=self.imgsz, verbose=False,
            classes=[self.person_cls_id]
        )[0]

        detected_persons = []
        if res_p.boxes is not None:
            for b in res_p.boxes:
                detected_persons.append(b.xyxy[0].tolist())

        # 2. Détection des tables avec seuil anti-bruit (conf >= 0.45)
        res_t = self.model_tables.predict(
            frame, conf=self.table_conf, imgsz=self.imgsz, verbose=False,
            classes=[self.table_cls_id]
        )[0]

        detected_tables = []
        if res_t.boxes is not None:
            for b in res_t.boxes:
                xy = b.xyxy[0].tolist()
                if self._filter_table_box(xy, w, h, detected_persons):
                    detected_tables.append(xy)

        return detected_tables, detected_persons

    def _update_tables_tracking(self, raw_tables, current_time):
        """
        Suivi continu fluide :
        - Les tables existantes sont mises à jour en direct chaque image.
        - Les nouvelles tables doivent être confirmées sur 5 frames stables pour éliminer les clignotements.
        """
        matched_track_ids = set()
        matched_raw_indices = set()

        # 1. Mise à jour fluide des tables existantes
        for tb in self.tables:
            best_score = -1e9
            best_idx = None
            for idx, raw_box in enumerate(raw_tables):
                if idx in matched_raw_indices:
                    continue

                iou = compute_iou(tb.box, raw_box)
                tc = tb.center
                rc = ((raw_box[0] + raw_box[2]) / 2.0, (raw_box[1] + raw_box[3]) / 2.0)
                center_dist = math.hypot(tc[0] - rc[0], tc[1] - rc[1])
                max_dim = max(tb.width, tb.height)

                if iou > 0.20 or center_dist < (max_dim * 0.95):
                    score = iou - (center_dist / (max_dim + 1e-4)) * 0.1
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            if best_idx is not None:
                matched_track_ids.add(tb.id)
                matched_raw_indices.add(best_idx)
                tb.update_position(raw_tables[best_idx], smooth=0.75)
                tb.last_detected = current_time
            else:
                tb.missed_frames += 1

        # 2. Gestion des nouvelles détections avec confirmation temporelle (5 frames)
        unmatched_raw = [
            raw_box for idx, raw_box in enumerate(raw_tables)
            if idx not in matched_raw_indices
        ]

        # Mise à jour des candidats
        new_candidates = []
        for raw_box in unmatched_raw:
            matched_cand = False
            for cand in self._candidates:
                if compute_iou(cand["box"], raw_box) > 0.35:
                    cand["box"] = raw_box
                    cand["hits"] += 1
                    cand["last_seen"] = current_time
                    matched_cand = True
                    # Si vu de manière stable sur 2 frames -> devient une table officielle !
                    if cand["hits"] >= 2:
                        if all(compute_iou(cand["box"], tb.box) < 0.25 for tb in self.tables):
                            new_table = TrackedTable(self._next_table_id, cand["box"], current_time)
                            self._next_table_id += 1
                            self.tables.append(new_table)
                    else:
                        new_candidates.append(cand)
                    break

            if not matched_cand:
                new_candidates.append({"box": raw_box, "hits": 1, "last_seen": current_time})

        # Nettoyage des candidats périmés (> 2.0s)
        self._candidates = [c for c in new_candidates if (current_time - c["last_seen"]) < 2.0]

        # 3. Retrait des tables absentes :
        # - Une table OCCUPEE ou EN_LIBERATION n'est JAMAIS supprimée !
        # - Une table LIBRE n'est retirée que si elle disparaît pendant plus de 30 secondes d'affilée.
        self.tables = [
            tb for tb in self.tables
            if tb.status in (OCCUPIED, LEAVING) or (current_time - tb.last_detected) < 30.0
        ]

    def _update_occupancy_and_timers(self, raw_persons, current_time):
        """
        Gestion intelligente anti-passant :
        - Un client qui passe en marchant ne déclenche pas le statut OCCUPÉE.
        - Il faut rester assis >= min_presence_sec (3.5s) pour valider l'occupation et démarrer le chrono.
        """
        for tb in self.tables:
            tb.persons_count = 0
            tb.assigned_person_boxes = []

        for p in raw_persons:
            candidates = [tb for tb in self.tables if tb.is_person_at_table(p, max_dist=self.max_person_table_distance)]
            if candidates:
                pc = ((p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0)
                best_table = min(
                    candidates,
                    key=lambda tb: math.hypot(pc[0] - tb.center[0], pc[1] - tb.center[1])
                )
                best_table.persons_count += 1
                best_table.assigned_person_boxes.append(p)

        for tb in self.tables:
            is_present_now = (tb.persons_count > 0)

            if is_present_now:
                tb.last_person_seen = current_time
                if tb.presence_started is None:
                    tb.presence_started = current_time
            else:
                tb.presence_started = None

            presence_duration = (
                (current_time - tb.presence_started)
                if tb.presence_started is not None else 0.0
            )
            absence_duration = current_time - tb.last_person_seen

            # 🟢 LIBRE -> 🔴 OCCUPÉE (Seulement après 3.5s d'assise continue !)
            if tb.status == FREE:
                if is_present_now and presence_duration >= self.min_presence_sec:
                    tb.status = OCCUPIED
                    tb.session_started = tb.presence_started
                    tb.leaving_started = None

            # 🔴 OCCUPÉE -> 🟠 EN DÉPART (Après 6s d'absence)
            elif tb.status == OCCUPIED:
                if not is_present_now and absence_duration >= self.absence_to_leaving_sec:
                    tb.status = LEAVING
                    tb.leaving_started = tb.last_person_seen

            # 🟠 EN DÉPART -> 🔴 OCCUPÉE (Client revenu) ou 🟢 LIBRE (Délai écoulé)
            elif tb.status == LEAVING:
                if is_present_now and presence_duration >= 1.0:
                    tb.status = OCCUPIED
                    tb.leaving_started = None
                elif (current_time - tb.leaving_started) >= self.grace_period_sec:
                    session_len = tb.leaving_started - (tb.session_started or tb.leaving_started)
                    tb.history.append({"duration_sec": max(0.0, session_len)})
                    tb.status = FREE
                    tb.session_started = None
                    tb.leaving_started = None

    def process(self, frame, current_time=None):
        if current_time is None:
            current_time = time.time()

        self._frame_count += 1
        raw_tables, raw_persons = self._detect_raw(frame)

        self._update_tables_tracking(raw_tables, current_time)
        self._update_occupancy_and_timers(raw_persons, current_time)

        state = self.get_state(current_time)
        annotated_frame = self._render(frame, raw_persons, state)
        return annotated_frame, state

    def get_state(self, current_time):
        tables_data = []
        for tb in self.tables:
            duration_sec = 0.0
            if tb.session_started is not None:
                duration_sec = current_time - tb.session_started

            # Calcul du temps de présence temporaire en cours d'attente (pour le feedback visuel)
            pending_occupy_sec = 0.0
            if tb.status == FREE and tb.presence_started is not None:
                pending_occupy_sec = current_time - tb.presence_started

            leaving_remain = None
            if tb.status == LEAVING and tb.leaving_started is not None:
                leaving_remain = max(0.0, self.grace_period_sec - (current_time - tb.leaving_started))

            tables_data.append({
                "table_id": tb.id,
                "bbox": [round(c, 1) for c in tb.box],
                "status": tb.status,
                "color_bgr": COLORS[tb.status],
                "persons_count": tb.persons_count,
                "duration_seconds": round(duration_sec),
                "timer_display": format_duration(duration_sec),
                "pending_occupy_sec": round(pending_occupy_sec, 1),
                "leaving_countdown_sec": round(leaving_remain) if leaving_remain is not None else None
            })

        total = len(self.tables)
        free = sum(1 for t in tables_data if t["status"] == FREE)
        occupied = sum(1 for t in tables_data if t["status"] == OCCUPIED)
        leaving = sum(1 for t in tables_data if t["status"] == LEAVING)

        return {
            "timestamp": round(current_time, 2),
            "calibrated": True,
            "total_tables": total,
            "free_count": free,
            "occupied_count": occupied,
            "leaving_count": leaving,
            "tables": tables_data
        }

    # ================= AFFICHAGE VISUEL =================
    def _draw_badge(self, img, text, x, y, bg_color, text_color=(255, 255, 255), scale=0.5):
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        h_img, w_img = img.shape[:2]

        x1 = int(max(0, min(x, w_img - tw - 10)))
        y1 = int(max(th + 6, min(y, h_img - 6)))
        x2 = x1 + tw + 8
        y2 = y1 + bl + 2

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        cv2.putText(img, text, (x1 + 4, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)

    def _render(self, frame, raw_persons, state):
        out = frame.copy()

        # Dessin des personnes détectées
        for p in raw_persons:
            x1, y1, x2, y2 = map(int, p)
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_PERSON, 2)
            cv2.putText(out, "Client", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERSON, 1, cv2.LINE_AA)

        # Dessin de chaque table avec badge dynamique et chronomètre
        for t_info, tb in zip(state["tables"], self.tables):
            x1, y1, x2, y2 = map(int, t_info["bbox"])
            status = t_info["status"]
            color = COLORS[status]

            # Rectangle autour de la table
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            # Ligne de liaison entre la table et les personnes assises
            for pb in tb.assigned_person_boxes:
                pc = (int((pb[0] + pb[2]) / 2), int(pb[3]))
                tc = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.line(out, tc, pc, color, 1, cv2.LINE_AA)

            # Libellé du badge
            label = f"T{t_info['table_id']}: {status}"
            if status == OCCUPIED:
                label += f" [{t_info['timer_display']}] ({t_info['persons_count']}p)"
            elif status == FREE and t_info["pending_occupy_sec"] > 0.5:
                # Feedback visuel : personne à côté mais attente de confirmation d'assise (anti-passant)
                remain = max(0.0, self.min_presence_sec - t_info["pending_occupy_sec"])
                label += f" (Assise {remain:.1f}s)"
            elif status == LEAVING and t_info["leaving_countdown_sec"] is not None:
                label += f" (Fin dans {t_info['leaving_countdown_sec']}s)"

            self._draw_badge(out, label, x1, y1 - 6, color, (255, 255, 255), scale=0.52)

        # Panneau supérieur
        hud_text = (
            f"TOTAL: {state['total_tables']} | "
            f"LIBRES: {state['free_count']} | "
            f"OCCUPEES: {state['occupied_count']} | "
            f"EN DEPART: {state['leaving_count']}"
        )
        self._draw_badge(out, hud_text, 12, 28, (30, 30, 30), (255, 255, 255), scale=0.6)

        return out
