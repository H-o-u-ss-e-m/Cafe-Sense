# ☕ Rapport Technique Détaillé : Smart Café IoT & Agent IA

Ce document est la version **détaillée et pédagogique** du rapport technique du projet **Smart Café** (café-restaurant *Coffee House Lac*, Tunis). Il reprend toutes les notions qui ont soulevé des questions (ONNX, Fine-Tuning, LoRA, RAG, configuration d'entraînement, etc.) et les explique au même endroit, avec :

- la **logique** de chaque choix technique ;
- la **signification exacte** de chaque paramètre ;
- le **déroulement réel** des deux entraînements, **vérifié à partir des deux notebooks Colab** (code *et* sorties d'exécution).

À partir de la **section 4 (Architecture Logicielle Globale)**, le contenu du rapport d'origine est conservé tel quel.

---

## 📑 Sommaire

1. [Vue d'Ensemble & Spécifications](#1-vue-densemble--spécifications)
2. [Notions fondamentales à comprendre (valables pour les deux modèles)](#2-notions-fondamentales-à-comprendre-valables-pour-les-deux-modèles)
   - [2.1 Pré-entraînement, Transfer Learning et Fine-Tuning](#21-pré-entraînement-transfer-learning-et-fine-tuning)
   - [2.2 Fine-Tuning classique vs LoRA](#22-fine-tuning-classique-vs-lora)
3. [Modèle 1 : Vision par Ordinateur IoT (YOLOv8n)](#3-modèle-1--vision-par-ordinateur-iot-yolov8n)
   - [3.1 Architecture du modèle YOLOv8n](#31-architecture-du-modèle-yolov8n)
   - [3.2 Dataset : de la vidéo aux annotations](#32-dataset--de-la-vidéo-aux-annotations)
   - [3.3 Entraînement sous Google Colab : déroulement réel étape par étape](#33-entraînement-sous-google-colab--déroulement-réel-étape-par-étape)
   - [3.4 La configuration d'entraînement expliquée paramètre par paramètre](#34-la-configuration-dentraînement-expliquée-paramètre-par-paramètre)
   - [3.5 Résultats réellement obtenus](#35-résultats-réellement-obtenus)
   - [3.6 Algorithme d'occupation des tables (CafeMonitorV2)](#36-algorithme-doccupation-des-tables-cafemonitorv2)
   - [3.7 Export ONNX / NCNN : pourquoi et est-ce toujours mieux ?](#37-export-onnx--ncnn--pourquoi-et-est-ce-toujours-mieux-)
4. [Modèle 2 : Agent IA Conversationnel (Qwen2.5-1.5B + LoRA)](#4-modèle-2--agent-ia-conversationnel-qwen25-15b--lora)
   - [4.1 Contrainte académique : zéro API commerciale](#41-contrainte-académique--zéro-api-commerciale)
   - [4.2 Le modèle de base Qwen2.5-1.5B-Instruct](#42-le-modèle-de-base-qwen25-15b-instruct)
   - [4.3 Même logique que YOLO : comment adapter Qwen à notre cas](#43-même-logique-que-yolo--comment-adapter-qwen-à-notre-cas)
   - [4.4 Le dataset d'entraînement](#44-le-dataset-dentraînement)
   - [4.5 Fine-Tuning LoRA : configuration expliquée paramètre par paramètre](#45-fine-tuning-lora--configuration-expliquée-paramètre-par-paramètre)
   - [4.6 Paramètres d'entraînement expliqués](#46-paramètres-dentraînement-expliqués)
   - [4.7 Déroulement réel du notebook Qwen, étape par étape](#47-déroulement-réel-du-notebook-qwen-étape-par-étape)
   - [4.8 Résultats réels et limites constatées](#48-résultats-réels-et-limites-constatées)
   - [4.9 Pourquoi l'adaptateur ne pèse que 73,9 Mo](#49-pourquoi-ladaptateur-ne-pèse-que-739-mo)
   - [4.10 Le RAG et le moteur hybride : rôle dans l'agent](#410-le-rag-et-le-moteur-hybride--rôle-dans-lagent)
   - [4.11 Résumé : comment l'agent est entraîné et utilisé](#411-résumé--comment-lagent-est-entraîné-et-utilisé)
5. [Tableau comparatif YOLO vs Qwen](#5-tableau-comparatif-yolo-vs-qwen)
6. [Écarts constatés entre le rapport initial et les notebooks](#6-écarts-constatés-entre-le-rapport-initial-et-les-notebooks)
7. [Architecture Logicielle Globale (Clean Architecture & Full-Stack)](#7-architecture-logicielle-globale-clean-architecture--full-stack) *(conservé du rapport d'origine)*
8. [Organisation des Fichiers & Déploiement](#8-organisation-des-fichiers--déploiement)
9. [Synthèse Technique](#9-synthèse-technique)

---

## 1. Vue d'Ensemble & Spécifications

Le projet **Smart Café** répond à une double problématique en restauration moderne :

1. **Gestion automatisée de la salle (IoT & Vision)** : suivre en temps réel l'occupation des tables sans capteurs physiques, grâce à une caméra (webcam PC ou caméras IP `CAM 1`, `CAM 2`, `CAM 3`) et un réseau de neurones convolutif.
2. **Expérience client personnalisée (Agent IA)** : un agent conversationnel autonome capable de renseigner les clients sur le menu (68 articles, formules, allergènes, prix en Dinars Tunisiens) et sur la disponibilité immédiate des tables d'après les caméras.

Le système repose sur **deux modèles d'IA pré-entraînés, spécialisés par Fine-Tuning** :

| | Modèle 1 – Vision | Modèle 2 – Agent IA |
|---|---|---|
| Modèle de base | YOLOv8n | Qwen2.5-1.5B-Instruct |
| Type de données | Images | Texte (conversations) |
| Mission | Détecter personnes et tables | Comprendre et répondre aux clients |
| Technique | Fine-Tuning (Transfer Learning) | Fine-Tuning **LoRA** (PEFT) |
| Livrable | `best.pt` → `best.onnx` | Adaptateur LoRA `adapter_model.safetensors` |
| Matériel d'entraînement | Google Colab, GPU Tesla T4 | Google Colab, GPU Tesla T4 |

> 💡 **Point clé** : on n'a pas créé deux IA *à partir de zéro*. On a pris deux modèles qui savent déjà faire beaucoup de choses, et on les a **spécialisés** pour le Smart Café.

```
                              ┌──────────────────────────────────────────────┐
                              │  Sources Vidéo : Webcam PC ou CAM1/CAM2/CAM3 │
                              └──────────────────────┬───────────────────────┘
                                                     │ Flux vidéo brut (MJPEG)
                                                     ▼
┌───────────────────────────────┐          ┌───────────────────────────────────┐
│        Client Web UI          │          │        Module Vision (YOLOv8)     │
│   (Angular 21 + Signals)      │◄─────────┤  - Détection tables & personnes   │
│  - Dashboard Tables en direct │          │  - Filtrage distance stricte <=30 │
│  - Sélecteur 4 Caméras        │          │  - États : LIBRE, OCCUPÉE, DÉPART │
│  - Catalogue & Filtres Menu   │          └─────────────────┬─────────────────┘
│  - Chatbot IA conversationnel │                            │
└───────────────┬───────────────┘                            │ État des tables (JSON)
                │                                            ▼
                │ Question client                  ┌───────────────────────────────────┐
                ▼                                  │      Agent Conversationnel IA     │
┌───────────────────────────────┐                  │  - LLM Qwen2.5-1.5B (LoRA Colab)  │
│     Serveur Backend Flask     ├─────────────────►│  - Moteur RAG & Base JSON Menu    │
│      (Clean Architecture)     │                  │  - Réponse bilingue (FR / Tunisien)│
│  - Routes REST & MJPEG        │◄─────────────────┤  - Connecté aux caméras direct    │
└───────────────────────────────┘    Réponse IA    └───────────────────────────────────┘
```

---

## 2. Notions fondamentales à comprendre (valables pour les deux modèles)

### 2.1 Pré-entraînement, Transfer Learning et Fine-Tuning

**Le Fine-Tuning consiste à prendre un modèle qui a déjà appris des choses et à continuer son apprentissage pour l'adapter à un besoin précis.**

Analogie : un étudiant possède déjà des connaissances générales. On veut qu'il devienne spécialiste d'un domaine. Au lieu de tout lui réapprendre depuis zéro, on lui donne des exemples spécialisés pour qu'il adapte ses connaissances.

```
Modèle pré-entraîné          Dataset spécialisé            Fine-Tuning              Modèle spécialisé
(connaissances générales) ─► (exemples de notre domaine) ─► (ajuste ses paramètres) ─► (adapté à Smart Café)
```

Le mot **Transfer Learning** désigne la même idée : on *transfère* ce qui a été appris sur une grande base de données (par exemple COCO pour YOLO) vers une nouvelle tâche plus petite.

**Pourquoi c'est indispensable ici ?**
- Entraîner un modèle depuis zéro demande des millions d'images ou des milliards de mots, et des GPU très coûteux.
- Avec le Fine-Tuning, quelques centaines d'images ou d'exemples de conversations suffisent, sur un simple GPU gratuit de Colab.

### 2.2 Fine-Tuning classique vs LoRA

| | Fine-Tuning classique | Fine-Tuning LoRA |
|---|---|---|
| Poids du modèle de base | Tous potentiellement modifiés | **Gelés** (non modifiés) |
| Ce qui est entraîné | Tout le réseau | De petites matrices ajoutées (A et B) |
| Mémoire GPU | Élevée | Faible |
| Fichier produit | Modèle complet | Petit adaptateur |
| Utilisé dans le projet pour | **YOLOv8n** (3,2 M de paramètres, petit modèle) | **Qwen2.5** (1,5 milliard de paramètres, gros modèle) |

Règle simple :
- **Fine-Tuning** = le *processus* qui adapte un modèle pré-entraîné à une tâche.
- **LoRA** = une *technique* pour réaliser ce Fine-Tuning en n'entraînant qu'une petite fraction de paramètres supplémentaires.

---

## 3. Modèle 1 : Vision par Ordinateur IoT (YOLOv8n)

### 3.1 Architecture du modèle YOLOv8n

Le modèle retenu est **YOLOv8n (Nano)** d'Ultralytics :

- **Paramètres** : ~3,2 millions (3 132 950 après fusion des couches, d'après le notebook), poids de ~6,2 à 6,5 Mo.
- **Backbone** : CSPDarknet53 modifié avec des blocs C2f (Cross-Stage Partial avec 2 convolutions) pour extraire rapidement des caractéristiques à plusieurs échelles.
- **Neck** : PANet (Path Aggregation Network), qui fusionne l'information sémantique profonde et les détails spatiaux de surface.
- **Head** : *anchor-free* (sans boîtes d'ancrage prédéfinies), ce qui réduit le calcul à l'inférence.

**Pourquoi la version « Nano » ?** Le notebook l'explique : un modèle lourd (v8x) serait inutilisable sur le CPU d'un Raspberry Pi (~0,5 FPS). YOLOv8n offre le meilleur compromis vitesse/précision pour un déploiement embarqué.

### 3.2 Dataset : de la vidéo aux annotations

#### Source des images
Le notebook utilise une vidéo de café `cafe.mp4` chargée depuis Google Drive :
- Résolution **640×352**, **30 FPS**, **110 668 frames** (**61,5 minutes**).
- Extraction d'**une image toutes les 2 secondes** (`step = fps × 2`).
- Seuls les **75 % premiers** de la vidéo sont utilisés pour l'entraînement ; les 25 % restants sont **réservés au test final** (jamais vus à l'entraînement).
- Résultat : **1 384 images extraites**.

#### Annotation automatique (modèle « enseignant »)
Annoter 1 384 images à la main serait très long. On a donc utilisé un **modèle enseignant** déjà très puissant, `yolo11x.pt` (pré-entraîné sur COCO), pour générer automatiquement les annotations. C'est ce qu'on appelle le **pseudo-labelling**.

Correspondance des classes COCO → classes du projet :

| Classe COCO | ID COCO | Classe Smart Café | ID |
|---|---|---|---|
| person | 0 | `person` | 0 |
| dining table | 60 | `table` | 1 |

**Filtres appliqués pour éviter les erreurs :**
1. **Confiance ≥ 0,35** (un seuil très bas comme 0,15 faisait détecter le sol).
2. **Surface de la boîte** entre 0,3 % et 20 % de l'image (une table n'est ni un petit débris ni toute la pièce).
3. **Ratio largeur/hauteur réaliste** entre 0,4 et 2,8.

**Pourquoi ces filtres ?** Le parquet en bois avec les reflets des lampes ressemble, vu de haut, à une table ronde en bois. Sans filtre géométrique, des zones du sol étaient étiquetées « table » et le réseau apprenait cette erreur.

**Résultat de l'annotation :** 5 737 tables et 5 524 personnes annotées sur 1 384 images.

> ⚠️ Le notebook évoque aussi des « images de fond négatives » (fichiers `.txt` vides). Dans l'exécution réelle, Ultralytics a affiché **« 0 backgrounds »** : toutes les images contenaient au moins un objet, donc ce mécanisme n'a en pratique pas été utilisé.

#### Contrôle visuel
Avant l'entraînement, 4 images aléatoires sont affichées avec leurs boîtes (vert = table, orange = personne) pour vérifier qu'**aucune boîte n'est posée sur le sol vide**.

#### Construction du dataset
- Séparation **chronologique** : **80 % Train = 1 107 images**, **20 % Val = 277 images**.
- Fichier `data.yaml` :

```yaml
path: /content/cafe_project/dataset
train: images/train
val: images/val
names:
  0: person
  1: table
```

### 3.3 Entraînement sous Google Colab : déroulement réel étape par étape

Notebook : `Entrainement_Modele_Cafe_Colab.ipynb`.

| Étape | Travail réalisé | Résultat constaté |
|---|---|---|
| 1 | Vérification GPU et installation (`ultralytics`, `opencv-python`, `onnx`, `onnxruntime`) | Ultralytics 8.4.208, GPU **Tesla T4** |
| 2 | Chargement de `cafe.mp4` depuis Google Drive | Vidéo copiée |
| 3 | Extraction des images (1 toutes les 2 s, sur 75 % de la vidéo) | **1 384 images** |
| 4 | Annotation automatique avec YOLO11x + filtres | 5 737 tables, 5 524 personnes |
| 5 | Contrôle visuel des annotations | 4 images vérifiées |
| 6 | Construction du dataset et `data.yaml` | 1 107 train / 277 val |
| 7 | **Fine-Tuning de YOLOv8n** | 44 époques, meilleur modèle à l'époque 32 |
| 8 | Export **ONNX** et **NCNN** (480×480) | `best.onnx` 12,1 Mo, NCNN 12,1 Mo |
| 9 | Test sur la partie jamais vue de la vidéo (300 frames) avec `CafeMonitorV2` | Vidéo `resultat_test.mp4` générée |
| 10 | Téléchargement des modèles | `best.pt` + `best.onnx` |

**Que fait YOLO pendant le Fine-Tuning ?** À chaque itération :
1. Il regarde un lot d'images.
2. Il prédit la position des personnes et des tables.
3. Il compare ses prédictions aux annotations.
4. Il calcule ses erreurs (les *loss*).
5. Il ajuste ses poids pour réduire ces erreurs.
6. Il recommence sur le lot suivant, puis sur toutes les époques.

Résultat : il adapte ses connaissances visuelles générales (COCO) à **ton** café, avec **ta** caméra et **ton** angle de vue.

### 3.4 La configuration d'entraînement expliquée paramètre par paramètre

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Transfer learning depuis COCO

results = model.train(
    data=f"{dataset_dir}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=12,
    save=True,
    project="/content/cafe_project/runs",
    name="cafe_model_v2",
    exist_ok=True,
    flipud=0.0
)
```

| Paramètre | Valeur | Signification exacte | Pourquoi ce choix |
|---|---|---|---|
| `YOLO("yolov8n.pt")` | modèle Nano pré-entraîné COCO | Point de départ : poids déjà appris sur COCO, qui connaît déjà « personne » et « table » | **Transfer Learning** : bien plus rapide et précis que partir de zéro |
| `data` | `data.yaml` | Fichier qui indique où sont les images train/val et les noms des classes | Dit à YOLO *quoi* apprendre |
| `epochs` | 50 | Nombre maximal de passages complets sur tout le dataset d'entraînement | Limite haute ; l'arrêt anticipé peut s'activer avant |
| `imgsz` | 640 | Les images sont redimensionnées en 640×640 pixels avant d'entrer dans le réseau | Taille standard de YOLO, bon compromis détail/vitesse |
| `batch` | 16 | Nombre d'images traitées ensemble avant une mise à jour des poids | 1 107 images ÷ 16 = **70 itérations par époque** (confirmé dans la sortie : `70/70`) |
| `patience` | 12 | **Early stopping** : arrêt si aucune amélioration de la validation pendant 12 époques | Évite le **surapprentissage** et gagne du temps |
| `save` | True | Sauvegarde les poids (`best.pt` = meilleur, `last.pt` = dernier) | Pour récupérer le meilleur modèle |
| `project` / `name` | `runs` / `cafe_model_v2` | Dossier et nom de l'expérience | Organisation des résultats |
| `exist_ok` | True | Autorise à écraser le dossier existant | Permet de relancer sans erreur |
| `flipud` | 0.0 | Probabilité de **retourner l'image verticalement** pendant l'augmentation | Une caméra de surveillance n'est jamais à l'envers : ce serait irréaliste |

**Paramètres appliqués automatiquement par Ultralytics** (visibles dans la sortie du notebook) :

| Paramètre | Valeur observée | Signification |
|---|---|---|
| `optimizer=auto` | **AdamW**, lr = 0,001667 | Algorithme qui met à jour les poids ; ici choisi automatiquement (le `lr0=0.01` par défaut est alors ignoré) |
| `amp=True` | activé | *Automatic Mixed Precision* : calculs en 16 bits partout où c'est sûr → moins de mémoire, plus rapide |
| `mosaic=1.0` | activé | Augmentation : colle 4 images en une pour varier les contextes |
| `close_mosaic=10` | 10 | Désactive la mosaïque pendant les 10 dernières époques pour affiner sur des images « normales » |
| `fliplr=0.5` | 50 % | Retournement horizontal aléatoire (gauche/droite), plausible pour un café |
| `hsv_h/s/v` | 0,015 / 0,7 / 0,4 | Variations de teinte, saturation, luminosité (changements d'éclairage) |
| `box` / `cls` / `dfl` | 7,5 / 0,5 / 1,5 | Poids des trois pertes ci-dessous |

**Les trois pertes (loss) affichées pendant l'entraînement :**
- `box_loss` : erreur sur la **position/taille** des boîtes.
- `cls_loss` : erreur sur la **classe** (person ou table ?).
- `dfl_loss` : *Distribution Focal Loss*, qui affine la précision des bords des boîtes.

Elles ont toutes diminué : par exemple `box_loss` de **0,876** (époque 1) à **0,241** (époque 44) et `cls_loss` de **1,068** à **0,261**. Le modèle apprend donc bien.

**Les métriques de validation, expliquées :**
- **Precision (P)** : parmi tout ce que le modèle a détecté, quelle part était correcte ? (peu de fausses alertes).
- **Recall (R)** : parmi tout ce qui existait réellement, quelle part a été trouvée ? (peu d'oublis).
- **IoU** : recouvrement entre la boîte prédite et la boîte réelle (surface commune ÷ surface totale).
- **mAP50** : précision moyenne en considérant une détection correcte si IoU ≥ 0,5.
- **mAP50-95** : moyenne sur des seuils IoU de 0,5 à 0,95, beaucoup plus exigeante (mesure la précision des boîtes).

### 3.5 Résultats réellement obtenus

D'après la sortie du notebook :

| Indicateur | Valeur |
|---|---|
| Époques exécutées | **44** sur 50 |
| Meilleure époque | **32** (arrêt anticipé : 32 + 12 = 44) |
| Durée d'entraînement | 0,107 h (≈ 6 min 30) |
| Mémoire GPU utilisée | ≈ 2,4 Go |
| mAP50 global | **98,9 %** |
| mAP50-95 global | **93,7 %** |
| Precision / Recall global | 97,8 % / 97,7 % |
| Vitesse sur T4 | 1,2 ms d'inférence + 0,1 ms prétraitement + 1,7 ms post-traitement par image |

| Classe | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| person | 277 | 1 056 | 96,0 % | 95,5 % | 98,4 % | 91,4 % |
| table | 277 | 1 333 | 99,5 % | 99,8 % | 99,5 % | 96,0 % |

> 📌 Ces résultats sont meilleurs que l'ancienne mention « > 92 % » du rapport initial. Voir la section 6 pour les précautions d'interprétation (données issues d'une seule vidéo, annotations automatiques).

### 3.6 Algorithme d'occupation des tables (CafeMonitorV2)

Détecter des boîtes ne suffit pas à savoir si une table est occupée. La classe `CafeMonitorV2` ajoute des règles géométriques et temporelles.

#### Version du notebook (moteur de test)
- **Calibration** : sur les 30 premières frames, les boîtes de tables sont regroupées par IoU > 0,4 ; une table est validée si elle apparaît dans ≥ 45 % des frames → positions de tables **stables**.
- **Zone d'assise** : boîte de la table élargie (+50 % en largeur, +40 % vers le haut, +80 % vers le bas) ; une personne est associée à la table dont le centre est le plus proche.
- **États** : 🟢 `LIBRE` → 🔴 `OCCUPÉE` (chronomètre `HH:MM:SS`) → 🟠 `EN_LIBERATION` si plus personne depuis **8 s** → 🟢 `LIBRE` après **60 s** de délai de grâce.

#### Version finale du projet (règles anti-erreurs renforcées)

**1. Règle stricte de distance client ↔ table.**
Un seuil trop large (80 % de la largeur de table, soit > 140 px) reliait par erreur des personnes éloignées : une cliente au comptoir (113 px de la table T2) marquait T2 comme occupée, et un musicien en arrière-plan (230 px) était relié à une table de devant.

La distance bord-à-bord minimale entre la boîte de la table `[x1, y1, x2, y2]` et celle de la personne `[px1, py1, px2, py2]` est :

$$dx = \max(0,\ x_1 - px_2,\ px_1 - x_2), \qquad dy = \max(0,\ y_1 - py_2,\ py_1 - y_2)$$
$$d_{\text{edge}} = \sqrt{dx^2 + dy^2}$$

Seuil d'admissibilité plafonné à 30 pixels :

$$d_{\text{max}} = \min\big(30.0,\ \max(12.0,\ \text{largeur} \times 0.18)\big)$$

Si $d_{\text{edge}} > d_{\text{max}}$, l'association est **rejetée** : la cliente au bar (113,7 px) et le musicien (236 px) ne sont plus jamais reliés, et la table reste 🟢 LIBRE.

```
[ Table T2 (Vide) ] -------- 113 px (REJETÉ car > 30 px) --------> [ Cliente au Bar ]
       │
       ▼
  🟢 Reste LIBRE
```

**2. Zone d'assise resserrée (+15 %).** La zone d'extension des chaises passe de +60 % à +15 % :

$$\text{Zone}_{\text{chaise}} = \left[ x_1 - 0.15 w,\ y_1 - 0.15 h,\ x_2 + 0.15 w,\ y_2 + 0.25 h \right]$$

**3. Filtre anti-filigrane d'angle (OSD).** Dans `CAM1.mp4`, le texte blanc « CAM 1 » en bas à droite était parfois détecté comme une petite table `T3`. Toute boîte collée au coin inférieur droit ($x_2 \ge W-20$ et $y_2 \ge H-20$) de surface < 5 % de l'image est rejetée.

**4. Filtre anti-passant (3 secondes).** Une présence continue ≥ 3,0 s est obligatoire pour démarrer le chronomètre d'occupation : quelqu'un qui traverse la salle ne déclenche pas « OCCUPÉE ».

> 🔧 Correction : la formule du rapport initial contenait une coquille (`py_3` au lieu de `py_2`), corrigée ci-dessus.

### 3.7 Export ONNX / NCNN : pourquoi et est-ce toujours mieux ?

#### Ce que fait ONNX
**ONNX ne réentraîne pas le modèle et ne lui apprend rien.** Il change seulement son **format d'exécution**.

- `best.pt` : modèle sauvegardé avec **PyTorch / Ultralytics** (format d'entraînement et de développement).
- `best.onnx` : le **même réseau**, exprimé dans un format ouvert standard (*Open Neural Network Exchange*), exécutable par **ONNX Runtime**.

```
best.pt (PyTorch / Ultralytics)
        │  Export ONNX
        ▼
best.onnx (même réseau, format standardisé)
        │
   ┌────┴─────────────┐
   ▼                  ▼
Serveur Windows    Raspberry Pi
(ONNX Runtime)     (ONNX Runtime)
```

Analogie : un logiciel écrit pour un environnement précis, qu'on veut faire tourner partout → on utilise un format d'échange compatible.

#### Différence entre `.pt` et `.onnx`

| Critère | `.pt` | `.onnx` |
|---|---|---|
| Format | PyTorch | Standard ONNX |
| Entraînement | Oui | **Non** : format d'inférence |
| Inférence CPU | Oui | Oui |
| GPU | CUDA possible | Plusieurs moteurs possibles |
| Déploiement | Dépendances PyTorch/Ultralytics | ONNX Runtime seulement |
| Compatibilité | Écosystème PyTorch | Plusieurs plateformes |

#### Pourquoi ONNX peut économiser des ressources
- Pas besoin d'installer tout PyTorch pour l'inférence (installation beaucoup plus légère).
- Moins de dépendances logicielles.
- Optimisations CPU (multithread) dans ONNX Runtime.
- Plusieurs *execution providers* selon le matériel.

> ⚠️ **ONNX n'est pas systématiquement plus rapide ou moins gourmand en mémoire.** Cela dépend du modèle, du matériel et des optimisations. À mesurer sur la machine cible.

#### Les paramètres d'export
```python
best_model = YOLO("best.pt")
best_model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
```

| Paramètre | Rôle |
|---|---|
| `format="onnx"` | Convertir le modèle PyTorch vers ONNX |
| `imgsz` | Résolution d'entrée du modèle exporté (640 dans le rapport initial ; **480** dans le notebook, recommandé pour Raspberry Pi car plus rapide) |
| `dynamic=False` | Fixe les dimensions d'entrée (au lieu de dimensions variables). Comme l'application utilise une résolution connue, cela facilite certaines optimisations |
| `simplify=True` | Simplifie le graphe de calcul : opérations redondantes éliminées ou fusionnées → inférence parfois plus rapide (le notebook utilise `onnxslim`) |

#### NCNN : l'export complémentaire pour Raspberry Pi
Le notebook exporte aussi au format **NCNN**, un moteur d'inférence C++ développé par Tencent, optimisé pour les processeurs **ARM** (Raspberry Pi).

#### Est-ce toujours mieux d'utiliser ONNX ?
**Non.** Voici comment choisir :

| Situation | Choix conseillé |
|---|---|
| Entraîner YOLO | `.pt` |
| Continuer le Fine-Tuning | `.pt` |
| Tester rapidement avec Ultralytics | `.pt` |
| Déployer sur serveur CPU | `.onnx` |
| Déployer sur Raspberry Pi (CPU) | `.onnx` ou NCNN, **à tester** |
| Déployer avec GPU NVIDIA | `.pt` ou ONNX avec CUDA |

#### Dans Smart Café
- **Colab (GPU T4)** : entraînement.
- **Serveur Windows ou Raspberry Pi** : utilisation du modèle sur les images caméra.
- On **n'entraîne pas** sur le Raspberry Pi : on exporte **une seule fois** `best.pt` en `best.onnx`, puis on prédit avec ONNX Runtime.

**À retenir :**
- `.pt` = format pratique pour développer, entraîner, modifier.
- `.onnx` = format pratique pour déployer dans différents environnements.
- **ONNX Runtime** = le moteur qui exécute le modèle ONNX.
- Meilleur choix pour l'architecture : **conserver les deux fichiers** (`best.pt` comme source, `best.onnx` pour le déploiement CPU).

---

## 4. Modèle 2 : Agent IA Conversationnel (Qwen2.5-1.5B + LoRA)

### 4.1 Contrainte académique : zéro API commerciale

- **Aucune dépendance externe payante** : pas d'appels à OpenAI (GPT-4), Anthropic (Claude) ou Google Gemini API.
- **Modèle embarqué local** : l'entraînement et l'inférence reposent exclusivement sur des bibliothèques open-source (`transformers`, `peft`, `torch`).

### 4.2 Le modèle de base Qwen2.5-1.5B-Instruct

- **Origine** : développé par Alibaba Cloud, reconnu comme l'un des meilleurs modèles compacts.
- **Architecture** : Transformer **décodeur auto-régressif** de **1,5 milliard de paramètres** (fichier de 3,09 Go en FP16). Il génère du texte *token par token*, en prédisant le prochain token à partir des précédents. Optimisé pour suivre des instructions, comprendre le contexte et le multilinguisme.
- **Contexte** : longueur native de 32 000 tokens, **limitée à 512 tokens** (`MAX_LENGTH = 512`) pour notre usage conversationnel (économie de mémoire).

**Ce que Qwen sait déjà** (pré-entraînement) :
- comprendre le langage et plusieurs langues ;
- générer des réponses et suivre des instructions ;
- tenir une conversation.

**Ce que Qwen ne sait PAS** :
- les tables, les prix, les plats de *Coffee House Lac* ;
- les règles d'occupation, les données de consommation ;
- l'état réel du café à l'instant présent.

### 4.3 Même logique que YOLO : comment adapter Qwen à notre cas

Oui, le principe est le même : on part d'un modèle pré-entraîné et on l'adapte par Fine-Tuning. La différence : YOLO travaille sur des **images**, Qwen sur du **texte**.

```
Qwen2.5-1.5B-Instruct          Dataset personnalisé           Fine-Tuning LoRA            Agent IA Smart Café
(connaissances générales) ──►  (questions/réponses du café) ──► (adaptation légère)  ──►   (modèle spécialisé)
```

**LoRA, en une phrase** : au lieu de modifier les 1,5 milliard de paramètres, on conserve les poids de base et on ajoute de **petites matrices entraînables** ; seules celles-ci apprennent à partir de nos données.

**Ce que le Fine-Tuning ne fait pas :** il ne donne pas à Qwen l'accès aux informations en temps réel. Pour savoir « combien de tables sont occupées ? », il faut connecter l'agent au backend / `CafeMonitorV2`.

| Élément | Rôle |
|---|---|
| Qwen + LoRA | Comprendre la question et construire une réponse |
| CafeMonitorV2 + Backend | Fournir les informations réelles du café |

### 4.4 Le dataset d'entraînement

Chaque exemple suit le **format conversationnel** de Qwen (rôles `system` / `user` / `assistant`) :

```python
{"messages": [
  {"role": "system",    "content": SYSTEM_PROMPT},
  {"role": "user",      "content": "Combien coûte le Capucin ?"},
  {"role": "assistant", "content": "Le Capucin est à 2.8 DT. Espresso avec une touche de lait mousseux tunisien. Allergènes : lait."}
]}
```

Le **SYSTEM_PROMPT** fixe l'identité de l'agent : agent du café *Coffee House Lac*, poli, comprenant français, anglais et dialecte tunisien, et ne donnant que des informations exactes basées sur le menu officiel.

**Génération du dataset à partir de `MENU_DATA`** (base JSON : 68 articles, 6 formules, infos pratiques) :

| Catégorie | Exemples | Génération |
|---|---|---|
| Salutations | « Bonjour ! », « 3aslema », « Merci beaucoup » | écrits à la main |
| Infos pratiques | wifi, parking, horaires, livraison, halal | écrits à la main |
| Disponibilité des tables | « Fama blasa tawwa ? » | écrits à la main |
| Produits | 3 questions par article : prix (FR), prix (derja : « 9adeh soum … ? »), ingrédients/détails | **générés automatiquement** en boucle sur le menu |
| Formules | « Parle-moi de la formule … », « 9adeh soum … ? » | générés en boucle |
| Allergènes & régimes | sans gluten, végétarien… | écrits à la main |

**Split** : 90 % entraînement / 10 % validation, dans l'ordre de génération (sans mélange).

> ⚠️ **Taille réelle du dataset : 104 exemples** (93 train + 11 validation) dans le notebook fourni. Le rapport initial mentionnait 391 dialogues (voir section 6).

### 4.5 Fine-Tuning LoRA : configuration expliquée paramètre par paramètre

#### Le principe mathématique
LoRA gèle les poids $W_0$ et apprend une correction de bas rang :

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}\,(B \cdot A)$$

Au lieu d'apprendre une énorme matrice $\Delta W$, on apprend deux petites matrices $A$ et $B$ dont le produit approche la modification utile.

```
Fine-Tuning classique          Fine-Tuning LoRA
1,5 milliard de paramètres     Poids de base GELÉS (modèle original conservé)
potentiellement entraînables   + petites matrices A et B entraînables
```

#### La configuration

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)
```

| Paramètre | Signification exacte |
|---|---|
| `r=16` | **Rang** : dimension intermédiaire des matrices A et B. Contrôle la *capacité* d'adaptation. Rang faible = adaptation légère ; rang élevé = plus de capacité mais plus de paramètres et de mémoire |
| `lora_alpha=32` | **Facteur d'échelle** : règle l'importance de la modification LoRA. Ici α/r = 32/16 = **2** : la contribution LoRA est multipliée par 2 |
| `lora_dropout=0.05` | **Dropout de 5 %** sur les entrées des couches LoRA pendant l'entraînement, pour éviter que le modèle ne dépende trop de certains détails des exemples (anti-surapprentissage) |
| `bias="none"` | On n'entraîne pas les termes de biais : seuls les paramètres LoRA sont entraînés |
| `task_type=CAUSAL_LM` | Indique à PEFT qu'on utilise un modèle de langage **auto-régressif** (prédit le prochain token) |
| `target_modules` | **Les couches où l'on insère LoRA** (voir ci-dessous) |

**Les 7 couches ciblées :**

| Couche | Rôle dans le Transformer |
|---|---|
| `q_proj` | **Query** : « que cherche-t-on dans le contexte ? » |
| `k_proj` | **Key** : représentation utilisée pour comparer et trouver l'information pertinente |
| `v_proj` | **Value** : l'information transportée par le mécanisme d'attention |
| `o_proj` | Projection de **sortie** de l'attention |
| `gate_proj` | Contrôle le passage de l'information dans le réseau Feed-Forward |
| `up_proj` | Projette vers une dimension intermédiaire plus grande |
| `down_proj` | Ramène vers la dimension initiale |

En ciblant l'attention **et** le Feed-Forward, LoRA adapte plusieurs parties importantes du Transformer.

#### Nombre de paramètres réellement entraînés

Sortie du notebook :

```
trainable params: 18,464,768 || all params: 1,562,179,072 || trainable%: 1.1820
```

Soit **seulement 1,18 %** des paramètres sont entraînés. Vérification par le calcul (Qwen2.5-1.5B : dimension cachée 1536, couche intermédiaire 8960, 28 couches, projections K/V de dimension 256) :

| Couche | Calcul r × (entrée + sortie) | Paramètres LoRA |
|---|---|---|
| q_proj | 16 × (1536 + 1536) | 49 152 |
| k_proj | 16 × (1536 + 256) | 28 672 |
| v_proj | 16 × (1536 + 256) | 28 672 |
| o_proj | 16 × (1536 + 1536) | 49 152 |
| gate_proj | 16 × (1536 + 8960) | 167 936 |
| up_proj | 16 × (1536 + 8960) | 167 936 |
| down_proj | 16 × (8960 + 1536) | 167 936 |
| **Total par couche** | | **659 456** |
| **× 28 couches** | | **18 464 768** ✅ |

### 4.6 Paramètres d'entraînement expliqués

```python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch",
    gradient_checkpointing=True,
    report_to="none",
    seed=42
)
```

| Paramètre | Valeur | Signification |
|---|---|---|
| `num_train_epochs` | 3 | Trois passages complets sur le dataset |
| `per_device_train_batch_size` | 2 | Deux exemples traités simultanément sur le GPU |
| `gradient_accumulation_steps` | 4 | Les gradients sont accumulés sur 4 batches avant de mettre à jour les poids |
| **Batch effectif** | **8** | 2 × 4 |
| `learning_rate` | 2e-4 (0,0002) | Amplitude des modifications à chaque mise à jour |
| `optim` | `adamw_torch` | Algorithme **AdamW** d'ajustement des paramètres |
| `fp16` | activé | Calculs en précision flottante 16 bits |
| `gradient_checkpointing` | activé | Économise la mémoire GPU en recalculant certaines activations |
| `eval_strategy` / `save_strategy` | `epoch` | Évaluation et sauvegarde à la fin de chaque époque |
| `load_best_model_at_end` | True | À la fin, recharge l'époque ayant la **meilleure `eval_loss`** |
| `save_total_limit` | 2 | Ne garde que 2 checkpoints (économie de disque) |
| `seed` | 42 | Reproductibilité |
| `MAX_LENGTH` | 512 | Longueur maximale (en tokens) de chaque exemple |

**Pourquoi un batch effectif de 8 ?**
1. Le GPU traite 2 exemples et calcule les gradients.
2. Il ne met **pas** à jour les poids tout de suite.
3. Il accumule les gradients sur 4 batches.
4. Il effectue alors **une** mise à jour.

→ On simule un batch de 8 sans charger 8 exemples en mémoire GPU simultanément.

**Pourquoi un learning rate de 2×10⁻⁴ ?** Trop élevé = apprentissage instable ; trop faible = très lent ; bien choisi = adaptation progressive. 2e-4 est une valeur courante pour LoRA (supérieure à celle d'un Fine-Tuning complet, car on n'entraîne que de petites matrices).

**Pourquoi FP16 ?** Au lieu de nombres sur 32 bits, une partie des calculs se fait sur 16 bits : moins de mémoire GPU et souvent des calculs plus rapides. Le modèle est chargé directement en `torch.float16`, ce qui évite aussi les problèmes de compilation de `bitsandbytes` (quantification) sur Colab.

**Pourquoi le Gradient Checkpointing ?** Pendant l'entraînement, le modèle garde en mémoire des résultats intermédiaires pour calculer les gradients. Le checkpointing en conserve moins et les **recalcule** au besoin : moins de mémoire, au prix de calculs supplémentaires.

**Tokenisation avec le chat template :** chaque conversation est transformée en texte avec `tokenizer.apply_chat_template(...)` (le format de balises propre à Qwen), puis tokenisée, tronquée et complétée à 512 tokens. `DataCollatorForLanguageModeling(mlm=False)` prépare un apprentissage *causal* : prédire chaque token à partir des précédents.

### 4.7 Déroulement réel du notebook Qwen, étape par étape

Notebook : `Entrainement_Agent_Cafe_Colab.ipynb`.

| Étape | Travail réalisé | Résultat constaté |
|---|---|---|
| 0 | Vérification du GPU | Tesla T4, 14,56 Go |
| 1 | Installation (`transformers`, `datasets`, `accelerate`, `peft`, `trl`) | OK (un avertissement de compatibilité `pyarrow`, sans impact) |
| 2 | Hyperparamètres globaux (`MAX_LENGTH=512`, batch 2, accumulation 4, 3 époques, lr 2e-4, seed 42) | — |
| 3 | Définition de `MENU_DATA` (68 articles, formules, infos) et sauvegarde `menu.json` | Menu sauvegardé |
| 4 | Génération du dataset de conversations | **104 exemples** |
| 5 | Split 90 % / 10 % | **93 train / 11 val** |
| 6 | Chargement de Qwen2.5-1.5B-Instruct en FP16 sur le GPU | 3,09 Go téléchargés, chargé sur CUDA |
| 7 | Injection de LoRA (PEFT) | 18 464 768 paramètres entraînables (1,18 %) |
| 8 | Tokenisation avec le chat template | Données prêtes |
| 9 | **Entraînement** (`Trainer.train()`) | **36 pas**, 143,5 s (≈ 2 min 24), loss moyenne d'entraînement **0,584** |
| 10 | Sauvegarde de l'adaptateur LoRA + tokenizer | `/content/cafe_agent_adapter` |
| 11 | Test en direct de 6 questions | Voir 4.8 |
| 12 | Archive ZIP et téléchargement | `cafe_agent_lora.zip` |

**Vérification du nombre de pas :** 93 exemples ÷ batch effectif 8 ≈ 12 mises à jour par époque × 3 époques = **36 pas** ✅.

### 4.8 Résultats réels et limites constatées

Test d'inférence (génération avec `temperature=0.3`, `top_p=0.9`, `repetition_penalty=1.1`) :

| Question | Réponse de l'agent | Verdict |
|---|---|---|
| Combien coûte le capucin ? | « 1.5 DT. Espresso concentré garni de thon frais… » | ❌ Prix faux (réel : 2,8 DT), description absurde |
| 9adeh soum l'express ? | « 3.5 DT. Espresso fort sans sucre ou lait. » | ❌ Prix faux (réel : 2,5 DT) |
| Allergique au gluten ? | Cite « Thon Fricassée, Pizza Margherita » | ❌ Ces plats contiennent du gluten |
| Formule pour deux personnes ? | « Formule Duo à 15.0 DT… » | ✅ Plausible |
| Code wifi ? | « Le WiFi est gratuit et couvre tout le café. » | ⚠️ Correct mais incomplet |
| Tables libres ? | « Oui, il y a encore quelques tables… » | ⚠️ Réponse générique, **aucun chiffre réel** |

**Ce que cela montre (point essentiel du rapport) :**
- Le Fine-Tuning a bien appris le **style** (ton poli, format de réponse, mélange français/derja).
- Mais avec un petit modèle et seulement 104 exemples, il **ne mémorise pas de façon fiable les faits** (prix, ingrédients, allergènes) et peut **halluciner**.
- C'est exactement la raison d'être de l'**architecture hybride** : les faits (prix, menu, tables) ne doivent pas venir de la mémoire de Qwen, mais d'un moteur de recherche dans le menu et du service de vision (section 4.10).

Pistes d'amélioration : plus d'exemples variés (reformulations, fautes de frappe, derja), mélange aléatoire avant le split, et fourniture au modèle du contexte récupéré (RAG) dans le prompt.

### 4.9 Pourquoi l'adaptateur ne pèse que 73,9 Mo

C'est l'un des grands avantages de LoRA. On ne conserve pas les 1,5 milliard de paramètres : on ne sauvegarde que les matrices A et B.

Cohérence avec le calcul : 18 464 768 paramètres × 4 octets (FP32) ≈ **73,9 Mo** ✅, à comparer aux 3,09 Go du modèle complet en FP16.

```
Qwen2.5-1.5B-Instruct (modèle original, 3,09 Go)
        +
adapter_model.safetensors (73,9 Mo)
        =
Agent adapté à Smart Café
```

> ⚠️ **L'adaptateur seul ne fonctionne pas.** Il nécessite le modèle Qwen de base compatible et sa configuration. Au démarrage de l'application, on charge Qwen, puis on lui applique l'adaptateur.

### 4.10 Le RAG et le moteur hybride : rôle dans l'agent

#### Fine-Tuning vs RAG : deux choses différentes
- **Fine-Tuning** : apprendre au modèle à mieux répondre dans le contexte du café (style, comportement).
- **RAG** : lui fournir, **au moment de la question**, des informations précises et à jour, **sans réentraîner** le modèle.

#### C'est quoi le RAG ?
**RAG = Retrieval-Augmented Generation** (« génération augmentée par recherche d'informations »).

1. Le client pose une question.
2. Le système **recherche** les informations pertinentes dans une source de données.
3. Il les **transmet au LLM** (Qwen).
4. Le LLM **formule** une réponse en s'appuyant sur ces informations.

Exemple : « Combien coûte le jus d'orange ? »

```
Question du client ─► Recherche dans le menu ─► LLM (Qwen) reçoit le prix réel ─► « Le jus d'orange frais coûte 4,5 DT. 🍊 »
```

Qwen **ne doit pas inventer** le prix à partir de ses connaissances : il reçoit le prix réel depuis le menu. Au vu des erreurs de la section 4.8, c'est indispensable.

#### Les quatre mécanismes du moteur hybride (`menu_engine.py`)

**A. Tolérance aux fautes de frappe et stop-words.** La question est nettoyée avant la recherche : *« jut »* → *« jus »*, *« desponible »* → *« disponible »*, *« donenr »* → *« donner »*. Sans cela, « jut d'orange » ne trouverait pas « jus d'orange ».

**B. Détection d'une catégorie complète.** « Je veux la liste des jus » / « fama sandwichs » : le système comprend qu'il s'agit d'un **groupe**, récupère tous les articles de la catégorie et renvoie la carte détaillée (prix, émoticônes, descriptions).

**C. Recherche ciblée d'un article.** « Combien coûte le capucin ? » : les mots de la question sont comparés au menu avec un score pondéré.

| Élément trouvé | Score |
|---|---|
| Correspondance avec le **nom** | +100 |
| Correspondance avec un **alias** | +80 |
| Correspondance avec les **ingrédients** | +25 |

L'article avec le meilleur score est retenu.

**D. Passerelle Vision IoT.** « Fama blasa ? » / « y a-t-il des tables libres ? » : cette fois on ne cherche pas dans le menu ; le système identifie une question de disponibilité et **interroge en direct le service de vision**, qui renvoie le nombre de tables libres issu de la caméra.

> 🔎 **Précision technique honnête.** La recherche dans le menu décrite ici est une **recherche locale pondérée** (noms, alias, ingrédients). C'est une forme simple de récupération d'informations dans une logique RAG, mais **pas nécessairement un RAG vectoriel** (embeddings + base vectorielle). L'appel au service de vision relève plutôt du **tool calling / appel d'API**. Pour affirmer qu'il s'agit d'un RAG « sémantique » au sens strict, il faudrait vérifier dans `menu_engine.py` l'usage d'embeddings et de similarité vectorielle.

#### Rôle exact de Qwen dans cette architecture
Qwen est surtout le **cerveau conversationnel** :
- il comprend l'intention du client ;
- il aide à déterminer quel module utiliser ;
- il reçoit les résultats du moteur de menu ou du service de vision ;
- il les transforme en réponse naturelle.

Exemple :
1. Client : « Fama blasa ? »
2. L'agent identifie une question sur les places disponibles.
3. Il interroge le service de vision → **3 tables libres**.
4. Qwen formule : « Oui ! Nous avons actuellement 3 tables libres. 😊 »

Le chiffre 3 vient du **service de vision**, pas de la mémoire de Qwen.

#### Ce qui est entraîné et ce qui ne l'est pas

| Élément | Comment fonctionne-t-il ? |
|---|---|
| Qwen | Fine-Tuning **LoRA** sur des conversations |
| Menu | Recherche **locale** dans produits, noms, alias, ingrédients |
| Correction des fautes | Règles et mécanismes dans `menu_engine.py` |
| Disponibilité des tables | Interrogation du **service de vision IoT** |
| RAG | Récupération d'informations pour **enrichir** la réponse de Qwen |

**Le RAG n'est pas un entraînement.** Exemple : si demain le jus d'orange passe de 5 DT à 6 DT :
- avec le Fine-Tuning seul, Qwen ne connaît **pas** le nouveau prix ;
- avec la recherche dans le menu, le système récupère directement le nouveau prix ;
- Qwen l'utilise pour répondre correctement.

#### Architecture globale de l'agent

```
Client (français ou dialecte tunisien)
        │
        ▼
Agent IA – Qwen + LoRA  (compréhension, gestion de la conversation)
        │
   ┌────┴───────────────┐
   ▼                    ▼
Menu Engine          Vision IoT
(correction,         (état des tables,
recherche produits)   places disponibles)
   └────────┬───────────┘
            ▼
Qwen génère la réponse à partir des résultats récupérés
            │
            ▼
      Réponse au client
```

### 4.11 Résumé : comment l'agent est entraîné et utilisé

Le modèle est **déjà pré-entraîné** : il connaît le langage. On ne part pas de zéro, on fait du **Fine-Tuning LoRA**.

1. **Charger le modèle pré-entraîné** : Qwen2.5-1.5B-Instruct et son tokenizer, en FP16 sur le GPU T4 de Colab.
2. **Préparer les données** : construire le dataset de conversations (system / user / assistant) à partir du menu et d'exemples manuels, puis split 90/10.
3. **Appliquer LoRA** : `r=16`, `alpha=32`, `dropout=0.05`, sur les 7 projections (q, k, v, o, gate, up, down) → 1,18 % de paramètres entraînables.
4. **Tokeniser** avec le chat template Qwen (longueur max 512).
5. **Entraîner** : 3 époques, batch 2, accumulation 4 (batch effectif 8), lr 2e-4, AdamW, FP16, gradient checkpointing, meilleure époque conservée selon `eval_loss`.
6. **Sauvegarder** l'adaptateur LoRA (≈ 73,9 Mo) et le tokenizer.
7. **Utiliser** dans l'application : charger Qwen + adaptateur ; l'agent traite les questions et collabore avec le moteur de menu et le service de vision.

> 🎤 **Phrase pour la présentation :** « Nous avons adapté Qwen2.5-1.5B-Instruct au domaine Smart Café grâce au Fine-Tuning LoRA sur un dataset de conversations. Nous avons ensuite développé une architecture hybride combinant un moteur local de recherche dans le menu et un service de vision IoT. La récupération d'informations permet d'enrichir les réponses du LLM avec des données réelles et actualisées, sans réentraîner le modèle à chaque changement de prix ou de disponibilité. »

---

## 5. Tableau comparatif YOLO vs Qwen

| Question | YOLOv8n (Vision) | Qwen2.5-1.5B + LoRA (Agent) |
|---|---|---|
| Modèle initial | YOLOv8n (COCO) | Qwen2.5-1.5B-Instruct |
| Nombre de paramètres | ~3,2 millions | ~1,56 milliard |
| Type de données | Images annotées | Conversations |
| Taille du dataset | 1 384 images (1 107 / 277) | 104 conversations (93 / 11) |
| Objectif | Détecter personnes et tables | Adapter style et réponses |
| Méthode | Fine-Tuning classique (Transfer Learning) | PEFT / LoRA |
| Paramètres modifiés | Poids du réseau | 18,5 M de paramètres LoRA (1,18 %) |
| Durée d'entraînement | ≈ 6,5 min (44 époques) | ≈ 2,4 min (3 époques, 36 pas) |
| Livrable | `best.pt` (6,2 Mo) | Adaptateur (73,9 Mo) + Qwen de base |
| Export de déploiement | ONNX / NCNN | Adaptateur + modèle de base |
| Accès aux données réelles | Analyse directe de la vidéo | Via menu engine et service de vision |

---

## 6. Écarts constatés entre le rapport initial et les notebooks

L'analyse du code et des sorties réelles fait apparaître des points à connaître pour la présentation et à corriger dans le rapport :

| Sujet | Rapport initial | Notebooks (réalité) |
|---|---|---|
| Taille du dataset Qwen | 391 dialogues | **104 exemples** (93 train + 11 val) |
| mAP50 YOLO | « > 92 % » | **98,9 %** global (mAP50-95 : 93,7 %) |
| Sources du dataset YOLO | CAM1, CAM2, CAM3 | Une seule vidéo `cafe.mp4` (640×352, 61,5 min) |
| Résolution d'export ONNX | 640, `dynamic=False` | **480**, `simplify=True` (+ export NCNN) |
| Vitesse CPU « ~45 ms » / GPU « ~12 ms » | Annoncées | Non mesurées dans le notebook (sur T4 : ~3 ms au total en validation) |
| Images de fond négatives | Mentionnées | **0 backgrounds** détectés par Ultralytics |
| Zones/seuils du moniteur | +15 %, 30 px, 3 s | Notebook : +50 %/+40 %/+80 %, départ 8 s, grâce 60 s (version de test) |
| VRAM ~5,2 Go (Qwen) | Annoncée | Non affichée dans les sorties (~2,4 Go pour YOLO) |

**Points techniques à examiner / à mentionner honnêtement :**
- **Annotations automatiques** : les labels viennent de YOLO11x (pseudo-labelling). Les métriques mesurent donc surtout l'accord avec ce modèle enseignant, pas une vérité terrain humaine.
- **Séparation Train/Val** : découpage chronologique d'une **même vidéo** ; des images voisines très ressemblantes peuvent gonfler les scores. Un vrai jeu de **test** indépendant (autre vidéo, autre jour) serait plus convaincant.
- **Qwen** : dataset petit et peu varié ; les validations ne remplacent pas un test sur des questions jamais vues. Les hallucinations de la section 4.8 justifient l'architecture hybride.
- **Moteur de test vs moteur final** : les règles décrites dans le rapport (30 px, 3 s…) correspondent à la version finale de `CafeMonitorV2` dans le projet, pas au code du notebook.

---

## 7. Architecture Logicielle Globale (Clean Architecture & Full-Stack)

Le projet est conçu selon les principes de la **Clean Architecture** (Séparation des responsabilités, indépendance des frameworks et testabilité).

```
projet_root/
├── backend/
│   └── app/
│       ├── core/           # Configuration globale (chemins, ports, CORS)
│       ├── domain/         # Entités métier pures (Table, MenuItem, ChatResponse)
│       ├── repositories/   # Accès aux données (MenuRepository sur menu.json)
│       ├── services/       # Logique applicative (VisionService, AgentService, MenuService)
│       ├── api/            # Contrôleurs HTTP Flask (routes_vision, routes_chat, routes_menu)
│       └── factory.py      # Factory pattern & injection CORS
├── frontend/
│   └── smart-cafe-app/
│       └── src/app/
│           ├── core/
│           │   ├── models/   # Interfaces TypeScript (table.model, menu.model, chat.model)
│           │   └── services/ # ApiService Angular (HttpClient)
│           ├── app.ts        # Composant racine réactif (Angular Signals)
│           ├── app.html      # Vue Dashboard, flux vidéo, menu & chatbot
│           └── app.css       # Design moderne Glassmorphism & thème sombre
└── test/                     # 🎥 Vidéos de surveillance réelles (CAM1, CAM2, CAM3)
```

### 7.1 Backend Flask (Clean Architecture & Multi-Sources)

- **`VisionService` :**
  - Gère un thread d'arrière-plan haute performance qui capture les frames vidéo, applique le modèle YOLO, annote les rectangles colorés (Vert/Rouge/Jaune) et encode les images en JPEG.
  - **Bascule Dynamique Multi-Sources (4 options) :**
    1. `0` : **📹 Caméra PC (Webcam Direct)**.
    2. `CAM1.mp4` : **🎥 Caméra 1 - Salle Principale** (`test/CAM1.mp4`).
    3. `CAM2.mp4` : **🎥 Caméra 2 - Terrasse** (`test/CAM2.mp4`).
    4. `CAM3.mp4` : **🎥 Caméra 3 - Comptoir & Entrée** (`test/CAM3.mp4`).
  - **Accélération DirectShow :** Utilise `cv2.CAP_DSHOW` sous Windows pour initialiser la caméra en moins de 150 ms sans freeze.
- **Routes REST & Streaming :**
  - `GET /video_feed` : Flux continu `multipart/x-mixed-replace; boundary=frame` (MJPEG sans latence).
  - `GET /api/status` : État JSON en direct (nombre de tables totales, libres, occupées, en départ).
  - `GET /api/vision/sources` & `POST /api/vision/source` : Liste et basculement à chaud des caméras.
  - `POST /api/chat` : Envoi d'une question à l'agent IA et réception de la réponse formatée.
  - `GET /api/menu` : Consultation du menu, filtres et formules.

### 7.2 Frontend Angular 21 (Signals & Reactive UI)

- **Architecture Réactive par Signals (`signal()`) :**
  - Évite les cycles de détection lourds traditionnels. Les signaux `roomStatus`, `videoStreamUrl`, `currentSource` et `chatMessages` répercutent instantanément chaque changement d'état.
- **Interface Utilisateur Moderne :**
  - Port d'écoute : **http://localhost:4201**.
  - Style Glassmorphism (effets de flou d'arrière-plan `backdrop-filter`, gradients néon cyan/ambre).
  - **Dashboard de surveillance :** 4 KPIs animés (Total, Libres, Occupées, En départ).
  - **Sélecteur 4 Caméras :** Boutons dédiés pour basculer en un clic entre la Webcam PC et les 3 caméras du café.
  - **Widget Chatbot IA Flottant :** Fenêtre de messagerie rétractable avec bulles de conversation et support complet du Markdown.
  - **Filtres Diététiques Dynamiques :** Filtrage instantané des articles sans gluten, végétariens ou par budget maximal (slider de prix).

### 7.3 Pipeline de Flux Vidéo MJPEG & Reconnexion Fluide

Pour pallier les interruptions lors du basculement d'une caméra à l'autre :
1. **Overlay avec Spinner de Rechargement :** Lorsqu'un basculement est initié, l'interface affiche un écran de transition stylisé avec indicateur rotatif (*"Reconnexion du flux en cours... Initialisation de la caméra..."*).
2. **Horodatage Anti-Cache :** L'URL du flux reçoit un paramètre de rafraîchissement unique (`?t=timestamp`), forçant le navigateur à ouvrir un socket propre sans afficher de frame gelée.
3. **Bouton Rafraîchir Manuel :** Intégré directement sur la carte vidéo pour réinitialiser le flux d'un clic en cas de micro-coupure matérielle.

---

## 8. Organisation des Fichiers & Déploiement

### Arborescence Nettoyée du Projet

```
Projet _ia/
├── 📁 ai_agent/                # Modèle conversationnel (LoRA Colab, menu.json, agent.py)
├── 📁 ai_vision/               # Modèles YOLO (best.pt, best.onnx, cafe_monitor_v2.py)
├── 📁 backend/                 # API Flask Clean Architecture (Port 5000)
├── 📁 frontend/                # Application Angular 21 (Port 4201)
├── 📁 test/                    # 🎥 Les 3 vidéos : CAM1.mp4, CAM2.mp4, CAM3.mp4
├── 📁 docs/                    # 📄 RAPPORT_SYSTEME_COMPLET.md et logo.png
├── 🚀 LANCER_SMART_CAFE.bat     # LE SEUL LANCEUR DU PROJET (Flask + Angular en 1 clic)
├── 📄 README.md                # Description générale du projet
└── 📦 requirements.txt         # Dépendances Python
```

### Lancement en 1 Clic sous Windows
Un unique fichier batch à la racine suffit :
👉 **Double-clic sur `LANCER_SMART_CAFE.bat`**

Ce script démarre automatiquement :
1. Le serveur Backend Flask sur **http://localhost:5000**.
2. L'application Web Frontend Angular sur **http://localhost:4201**.
3. Ouvre directement votre navigateur sur le tableau de bord interactif.

---

## 9. Synthèse Technique

| Composant | Technologie | Rôle Principal |
| :--- | :--- | :--- |
| **Détection d'Objets** | YOLOv8n (PyTorch / ONNX) | Détecte tables et personnes en temps réel (640x640, ~60 FPS) |
| **Logique Spatiale IoT** | Seuil strict $\le 30\text{ px}$ + Centroïde | Associe précisément les clients aux tables sans fausses liaisons à distance |
| **Sources Vidéo** | Webcam PC + CAM1, CAM2, CAM3 | Surveillance multi-zones (salle, terrasse, comptoir) avec basculement fluide |
| **Agent Conversationnel**| Qwen2.5-1.5B + LoRA (Colab) | Génère des réponses expertes et courtoises bilingues (FR / Tunisien) |
| **Moteur Sémantique Local**| RAG déterministe + JSON | Fournit des réponses instantanées (< 10 ms) et tolérantes aux fautes |
| **API & Streaming** | Python Flask (Clean Arch.) | Expose les routes REST et le flux vidéo MJPEG (DirectShow $<150\text{ ms}$) |
| **Interface Utilisateur** | Angular 21 (Signals) | Affiche le dashboard temps réel, le sélecteur 4 caméras et le chatbot |
