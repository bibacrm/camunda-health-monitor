"""
API routes - RESTful endpoints
"""
from flask import Blueprint, jsonify, current_app
from datetime import datetime
from helpers.error_handler import handle_errors
from helpers.db_helper import execute_query
from services.camunda_service import collect_engine_health
from services.database_service import collect_database_metrics

api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/health')
@handle_errors(context="Fetching cluster health")
def api_health():
    """API endpoint for full health data"""
    data = collect_engine_health()
    data['timestamp'] = data['timestamp'].isoformat()
    return jsonify(data)


@api_bp.route('/metrics/stuck-instances')
@handle_errors(context="Fetching stuck instances")
def api_stuck_instances():
    """Get count of stuck instances"""
    stuck_days = current_app.config.get('STUCK_INSTANCE_DAYS', 7)
    result = execute_query(f"""
        SELECT COUNT(*) as count 
        FROM (
            SELECT 1 
            FROM act_ru_execution pi 
            JOIN (
                SELECT proc_inst_id_, MAX(start_time_) as last_update 
                FROM act_hi_actinst 
                GROUP BY proc_inst_id_
            ) la ON pi.proc_inst_id_ = la.proc_inst_id_ 
            WHERE pi.parent_id_ IS NULL 
            AND la.last_update < NOW() - INTERVAL '{stuck_days} days'
        ) as stuck
    """)
    return jsonify({
        "value": result[0]['count'] if result else 0,
        "timestamp": datetime.now().isoformat()
    })


@api_bp.route('/metrics/pending-messages')
@handle_errors(context="Fetching pending messages")
def api_pending_messages():
    """Get count of pending message subscriptions"""
    result = execute_query("""
        SELECT COUNT(*) as count 
        FROM act_ru_event_subscr 
        WHERE event_type_ = 'message'
    """)
    return jsonify({
        "value": result[0]['count'] if result else 0,
        "timestamp": datetime.now().isoformat()
    })


@api_bp.route('/metrics/pending-signals')
@handle_errors(context="Fetching pending signals")
def api_pending_signals():
    """Get count of pending signal subscriptions"""
    result = execute_query("""
        SELECT COUNT(*) as count 
        FROM act_ru_event_subscr 
        WHERE event_type_ = 'signal'
    """)
    return jsonify({
        "value": result[0]['count'] if result else 0,
        "timestamp": datetime.now().isoformat()
    })


@api_bp.route('/metrics/job-throughput')
@handle_errors(context="Fetching job throughput")
def api_job_throughput():
    """Get job execution throughput (jobs per minute)"""
    result = execute_query("""
        SELECT COUNT(*) as count 
        FROM act_hi_job_log 
        WHERE job_state_=2 
        AND timestamp_ >= now() - interval '10 minutes'
    """)
    jobs_last_10min = result[0]["count"] if result else 0
    return jsonify({
        "jobs_executed_per_min": round(jobs_last_10min / 10.0, 1),
        "timestamp": datetime.now().isoformat()
    })


@api_bp.route('/metrics/database')
@handle_errors(context="Fetching database metrics")
def api_database_metrics():
    """Get database storage and performance metrics"""
    data = collect_database_metrics()
    data["timestamp"] = datetime.now().isoformat()
    return jsonify(data)

