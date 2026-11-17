"""
=============================================================================
☕ SMART CAFE - CORE CONFIGURATION (Clean Architecture)
=============================================================================
Centralise la configuration, les chemins absolus et les constantes système.
=============================================================================
"""

import os

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    AI_VISION_DIR = os.path.join(PROJECT_ROOT, "ai_vision")
    AI_AGENT_DIR = os.path.join(PROJECT_ROOT, "ai_agent")
    TEST_DIR = os.path.join(PROJECT_ROOT, "test")
    DATA_DIR = os.path.join(AI_AGENT_DIR, "data")
    MENU_JSON_PATH = os.path.join(DATA_DIR, "menu.json")
    
    # Vision & YOLO
    DEFAULT_SOURCE = "0"
    DEFAULT_VIDEO = os.path.join(TEST_DIR, "CAM1.mp4")
    DEFAULT_MODEL = os.path.join(AI_VISION_DIR, "models", "best.pt")
    TABLE_CONF = 0.35
    PERSON_CONF = 0.35
    IMGSZ = 640
    SKIP_FRAMES = 1
    
    # Server HTTP
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = False
    
    # CORS
    CORS_ORIGINS = ["http://localhost:4201", "http://127.0.0.1:4201", "http://localhost:4200", "http://127.0.0.1:4200", "*"]
