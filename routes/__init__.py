"""
Routes package for Camunda Health Monitor
"""
from flask import Blueprint

# Import blueprints
from .main import main_bp
from .api import api_bp
from .metrics import metrics_bp

# List of all blueprints
blueprints = [
    main_bp,
    api_bp,
    metrics_bp
]


def register_blueprints(app):
    """Register all blueprints with the Flask app"""
    for blueprint in blueprints:
        app.register_blueprint(blueprint)

