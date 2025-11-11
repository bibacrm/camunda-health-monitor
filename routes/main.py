"""
Main routes - Dashboard
"""
from flask import Blueprint, render_template, current_app
from services.camunda_service import collect_engine_health

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Main dashboard page"""
    data = collect_engine_health()
    stuck_days = current_app.config.get('STUCK_INSTANCE_DAYS', 7)
    return render_template('index.html', data=data, stuck_days=stuck_days)

