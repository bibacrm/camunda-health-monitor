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

    # UI Configuration - pass backend config to frontend
    ui_config = {
        'autoRefreshInterval': current_app.config.get('UI_AUTO_REFRESH_INTERVAL_MS', 30000),
        'archiveDaysThreshold': current_app.config.get('DB_ARCHIVE_THRESHOLD_DAYS', 90),
        'aiDisplayLimits': {
            'anomalies': current_app.config.get('AI_UI_RESULTS_LIMIT', 20),
            'incidents': current_app.config.get('AI_UI_RESULTS_LIMIT', 20),
            'bottlenecks': current_app.config.get('AI_UI_RESULTS_LIMIT', 20),
            'jobPredictions': current_app.config.get('AI_UI_RESULTS_LIMIT', 20),
            'leaderboard': current_app.config.get('AI_UI_RESULTS_LIMIT', 20)
        }
    }

    return render_template('index.html', data=data, stuck_days=stuck_days, ui_config=ui_config)

