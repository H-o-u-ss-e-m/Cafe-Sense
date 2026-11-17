"""
=============================================================================
☕ FLASK APPLICATION FACTORY (Clean Architecture)
=============================================================================
Assemble l'application Flask, initialise les services métier et enregistre
les blueprints d'API.
=============================================================================
"""

import os
from flask import Flask, render_template

from .core.config import Config
from .repositories.menu_repository import MenuRepository
from .services.vision_service import VisionService
from .services.agent_service import AgentService
from .services.menu_service import MenuService

from .api.routes_vision import vision_bp, init_vision_routes
from .api.routes_chat import chat_bp, init_chat_routes
from .api.routes_menu import menu_bp, init_menu_routes


def create_app(
    source=None,
    model_path=None,
    table_conf=Config.TABLE_CONF,
    person_conf=Config.PERSON_CONF,
    imgsz=Config.IMGSZ,
    skip_frames=Config.SKIP_FRAMES
) -> Flask:
    templates_dir = os.path.join(Config.BASE_DIR, "templates")
    app = Flask(__name__, template_folder=templates_dir)

    # Activer CORS pour Angular (http://localhost:4200) et tout client
    try:
        from flask_cors import CORS
        CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    except ImportError:
        @app.after_request
        def add_cors_headers(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return response

    # 1. Initialisation des Référentiels (Data Layer)
    menu_repo = MenuRepository()

    # 2. Initialisation des Services Métier (Use Cases Layer)
    vision_service = VisionService(
        source=source,
        model_path=model_path,
        table_conf=table_conf,
        person_conf=person_conf,
        imgsz=imgsz,
        skip_frames=skip_frames
    )
    agent_service = AgentService()
    menu_service = MenuService(repository=menu_repo)

    # Démarrer le traitement vidéo
    vision_service.start()

    # 3. Injection de dépendance dans les Contrôleurs (API Layer)
    init_vision_routes(vision_service)
    init_chat_routes(agent_service, vision_service)
    init_menu_routes(menu_service)

    # 4. Enregistrement des Blueprints
    app.register_blueprint(vision_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(menu_bp)

    # Route pour le dashboard web HTML legacy (fallback de secours)
    @app.route("/")
    def index():
        return render_template("index.html")

    return app
