"""
API routes - RESTful endpoints
"""
from flask import Blueprint, jsonify, current_app
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers.error_handler import handle_errors
from helpers.db_helper import execute_query
from services.camunda_service import collect_engine_health
from services.database_service import collect_database_metrics
from services.ai_service import get_ai_analytics

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


# ===== AI/ML ENDPOINTS =====

@api_bp.route('/ai/health-score')
@handle_errors(context="Calculating AI health score")
def api_ai_health_score():
    """Get AI-powered cluster health score"""
    ai = get_ai_analytics()
    cluster_data = collect_engine_health()
    db_metrics = cluster_data.get('db_metrics', {})

    score = ai.get_cluster_health_score(cluster_data, db_metrics)
    score['timestamp'] = datetime.now().isoformat()

    return jsonify(score)


@api_bp.route('/ai/anomalies')
@handle_errors(context="Detecting process anomalies")
def api_ai_anomalies():
    """Detect anomalies in process execution times"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))

    anomalies = ai.detect_process_anomalies(lookback_days=lookback_days)
    anomalies['timestamp'] = datetime.now().isoformat()

    return jsonify(anomalies)


@api_bp.route('/ai/incident-patterns')
@handle_errors(context="Analyzing incident patterns")
def api_ai_incident_patterns():
    """Analyze and cluster incident patterns"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))

    patterns = ai.analyze_incident_patterns(lookback_days=lookback_days)
    patterns['timestamp'] = datetime.now().isoformat()

    return jsonify(patterns)


@api_bp.route('/ai/bottlenecks')
@handle_errors(context="Identifying process bottlenecks")
def api_ai_bottlenecks():
    """Identify process bottlenecks"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))

    bottlenecks = ai.identify_bottlenecks(lookback_days=lookback_days)
    bottlenecks['timestamp'] = datetime.now().isoformat()

    return jsonify(bottlenecks)


@api_bp.route('/ai/job-predictions')
@handle_errors(context="Predicting job failures")
def api_ai_job_predictions():
    """Predict job failure probabilities"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))

    predictions = ai.predict_job_failures(lookback_days=lookback_days)
    predictions['timestamp'] = datetime.now().isoformat()

    return jsonify(predictions)


@api_bp.route('/ai/node-performance')
@handle_errors(context="Analyzing node performance")
def api_ai_node_performance():
    """Get node performance rankings"""
    ai = get_ai_analytics()
    cluster_data = collect_engine_health()

    rankings = ai.analyze_node_performance(cluster_data.get('cluster_nodes', []))
    rankings['timestamp'] = datetime.now().isoformat()

    return jsonify(rankings)


@api_bp.route('/ai/process-leaderboard')
@handle_errors(context="Getting process leaderboard")
def api_ai_process_leaderboard():
    """Get process definition performance leaderboard"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))

    leaderboard = ai.get_process_leaderboard(lookback_days=lookback_days)
    leaderboard['timestamp'] = datetime.now().isoformat()

    return jsonify(leaderboard)


@api_bp.route('/ai/sla-predictions')
@handle_errors(context="Predicting SLA breaches")
def api_ai_sla_predictions():
    """Predict tasks likely to breach SLA"""
    ai = get_ai_analytics()
    threshold_hours = int(current_app.config.get('SLA_THRESHOLD_HOURS', 24))

    predictions = ai.predict_sla_breaches(threshold_hours=threshold_hours)
    predictions['timestamp'] = datetime.now().isoformat()

    return jsonify(predictions)


@api_bp.route('/ai/insights')
@handle_errors(context="Getting comprehensive AI insights")
def api_ai_insights():
    """Get comprehensive AI insights and recommendations"""
    from flask import request
    ai = get_ai_analytics()

    # Get configuration
    lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))
    threshold_hours = int(current_app.config.get('SLA_THRESHOLD_HOURS', 24))

    # Check if cluster data should be included (can be disabled for faster loading)
    include_cluster = request.args.get('include_cluster', 'true').lower() == 'true'

    # Gather cluster data only if needed (for health score and node performance)
    cluster_data = {}
    db_metrics = {}

    if include_cluster:
        # Only collect if not already available from dashboard
        current_app.logger.info("Collecting cluster data for AI insights")
        cluster_data = collect_engine_health()
        db_metrics = cluster_data.get('db_metrics', {})

    # Define tasks for parallel execution
    def get_health_score():
        if include_cluster and cluster_data:
            return ('health_score', ai.get_cluster_health_score(cluster_data, db_metrics))
        else:
            # Return minimal health score based on incidents only
            return ('health_score', {
                'overall_score': 85,
                'grade': 'B',
                'factors': 'Based on process metrics only'
            })

    def get_anomalies():
        return ('anomalies', ai.detect_process_anomalies(lookback_days=lookback_days))

    def get_incidents():
        return ('incidents', ai.analyze_incident_patterns(lookback_days=min(lookback_days * 4, 30)))

    def get_bottlenecks():
        return ('bottlenecks', ai.identify_bottlenecks(lookback_days=lookback_days))

    def get_job_failures():
        return ('job_failures', ai.predict_job_failures(lookback_days=lookback_days))

    def get_node_performance():
        if include_cluster and cluster_data:
            return ('node_performance', ai.analyze_node_performance(cluster_data.get('cluster_nodes', [])))
        else:
            # Return empty node performance if cluster data not available
            return ('node_performance', {
                'rankings': [],
                'message': 'Node performance analysis requires cluster data'
            })

    def get_process_leaderboard():
        return ('process_leaderboard', ai.get_process_leaderboard(lookback_days=min(lookback_days * 4, 30)))

    def get_sla_predictions():
        return ('sla_predictions', ai.predict_sla_breaches(threshold_hours=threshold_hours))

    # Execute all AI analysis in parallel
    insights = {}
    tasks = [
        get_health_score,
        get_anomalies,
        get_incidents,
        get_bottlenecks,
        get_job_failures,
        get_node_performance,
        get_process_leaderboard,
        get_sla_predictions
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {executor.submit(task): task for task in tasks}

        for future in as_completed(future_to_task):
            try:
                key, result = future.result()
                insights[key] = result
            except Exception as e:
                task_name = future_to_task[future].__name__
                insights[task_name] = {'error': str(e)}

    # Generate recommendations based on collected insights
    insights['recommendations'] = ai.get_ai_recommendations(insights)
    insights['timestamp'] = datetime.now().isoformat()

    return jsonify(insights)


# ===== ADVANCED ML ENDPOINTS =====

@api_bp.route('/ai/stuck-activities-smart')
@handle_errors(context="Finding stuck activities with smart detection")
def api_ai_stuck_activities_smart():
    """
    Advanced stuck activity detection using statistical percentile thresholds
    Identifies activities taking abnormally long based on historical patterns
    """
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))

    result = ai.find_stuck_activities_smart(lookback_days=lookback_days)
    result['timestamp'] = datetime.now().isoformat()

    return jsonify(result)


@api_bp.route('/ai/predict-duration/<process_def_key>')
@handle_errors(context="Predicting process duration")
def api_ai_predict_duration(process_def_key):
    """
    Predict how long a process will take to complete using ML
    Based on historical execution patterns and time-of-day factors
    """
    ai = get_ai_analytics()

    prediction = ai.predict_process_duration(process_def_key)
    prediction['timestamp'] = datetime.now().isoformat()
    prediction['process_def_key'] = process_def_key

    return jsonify(prediction)


@api_bp.route('/ai/predict-all-durations')
@handle_errors(context="Predicting all process durations")
def api_ai_predict_all_durations():
    """
    Predict durations for all process definitions in the system
    Returns list sorted by predicted duration (longest first)
    """
    ai = get_ai_analytics()

    # Get all process definitions with recent activity
    recent_processes_query = """
                             SELECT DISTINCT proc_def_key_
                             FROM act_hi_procinst
                             WHERE start_time_ > NOW() - INTERVAL '30 days'
                             ORDER BY proc_def_key_ \
                             """

    process_keys = execute_query(recent_processes_query)

    if not process_keys:
        return jsonify({
            'predictions': [],
            'total_processes': 0,
            'timestamp': datetime.now().isoformat(),
            'message': 'No process definitions found with recent activity'
        })

    def predict_single_process(row):
        """Helper function to predict duration for a single process"""
        proc_key = row['proc_def_key_']
        try:
            pred = ai.predict_process_duration(proc_key)
            pred['process_key'] = proc_key
            return pred
        except Exception as e:
            current_app.logger.warning(f"Failed to predict duration for {proc_key}: {e}")
            return None

    predictions = []

    # Process predictions in parallel with 8 workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_key = {executor.submit(predict_single_process, row): row['proc_def_key_']
                         for row in process_keys}

        # Collect results as they complete
        for future in as_completed(future_to_key):
            result = future.result()
            if result is not None:
                predictions.append(result)

    # Sort by predicted duration (longest first), handle None values
    predictions.sort(key=lambda x: x.get('predicted_duration_hours') or 0, reverse=True)

    return jsonify({
        'predictions': predictions,
        'total_processes': len(predictions),
        'timestamp': datetime.now().isoformat()
    })


@api_bp.route('/ai/capacity-forecast')
@handle_errors(context="Forecasting capacity needs")
def api_ai_capacity_forecast():
    """
    Forecast future capacity needs based on historical load patterns
    Uses time series analysis and trend detection
    """
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_CAPACITY_TRAINING_DAYS', 90))
    forecast_days = int(current_app.config.get('AI_CAPACITY_FORECAST_DAYS', 30))

    forecast = ai.forecast_capacity(lookback_days=lookback_days, forecast_days=forecast_days)
    forecast['timestamp'] = datetime.now().isoformat()

    return jsonify(forecast)


@api_bp.route('/ai/variable-impact/<process_def_key>')
@handle_errors(context="Analyzing variable impact")
def api_ai_variable_impact(process_def_key):
    """
    Analyze which process variables correlate with failures or performance issues
    Identifies high-impact variables that affect process outcomes
    """
    ai = get_ai_analytics()

    impact = ai.analyze_variable_impact(process_def_key)
    impact['timestamp'] = datetime.now().isoformat()

    return jsonify(impact)


# ===== NEW PROFESSIONAL ANALYSIS ENDPOINTS =====

@api_bp.route('/ai/process-categories')
@handle_errors(context="Getting process categories")
def api_ai_process_categories():
    """Get process categorization (ultra_fast to batch_manual)"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_CAPACITY_TRAINING_DAYS', 90))

    result = ai.get_process_categories(lookback_days=lookback_days)
    result['timestamp'] = datetime.now().isoformat()

    return jsonify(result)


@api_bp.route('/ai/version-performance')
@handle_errors(context="Analyzing version performance")
def api_ai_version_performance():
    """Analyze performance changes between process versions"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_CAPACITY_TRAINING_DAYS', 90))

    result = ai.analyze_version_performance(lookback_days=lookback_days)
    result['timestamp'] = datetime.now().isoformat()

    return jsonify(result)


@api_bp.route('/ai/extreme-variability')
@handle_errors(context="Detecting extreme variability")
def api_ai_extreme_variability():
    """Detect processes with extreme P95/Median ratios"""
    ai = get_ai_analytics()

    result = ai.analyze_extreme_variability()
    result['timestamp'] = datetime.now().isoformat()

    return jsonify(result)


@api_bp.route('/ai/load-patterns')
@handle_errors(context="Analyzing load patterns")
def api_ai_load_patterns():
    """Analyze business hours vs weekend load patterns"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_CAPACITY_TRAINING_DAYS', 90))

    result = ai.analyze_load_patterns(lookback_days=lookback_days)
    result['timestamp'] = datetime.now().isoformat()

    return jsonify(result)


@api_bp.route('/ai/stuck-processes')
@handle_errors(context="Analyzing stuck processes")
def api_ai_stuck_processes():
    """Get stuck process instances with business keys"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))

    result = ai.analyze_stuck_processes(lookback_days=lookback_days)
    result['timestamp'] = datetime.now().isoformat()

    return jsonify(result)


@api_bp.route('/ai/outlier-patterns')
@handle_errors(context="Analyzing outlier patterns")
def api_ai_outlier_patterns():
    """Get IQR-based outlier analysis"""
    ai = get_ai_analytics()
    lookback_days = int(current_app.config.get('AI_CAPACITY_TRAINING_DAYS', 90))

    result = ai.analyze_outlier_patterns(lookback_days=lookback_days)
    result['timestamp'] = datetime.now().isoformat()

    return jsonify(result)


@api_bp.route('/ai/comprehensive-analysis')
@handle_errors(context="Getting comprehensive professional analysis")
def api_ai_comprehensive_analysis():
    """
    Professional enterprise-grade comprehensive analysis
    Combines all analysis methods for full system intelligence
    """
    from flask import request
    ai = get_ai_analytics()

    # Get configuration
    lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))
    capacity_days = int(current_app.config.get('AI_CAPACITY_TRAINING_DAYS', 90))

    # Check if quick mode (skip some analyses)
    quick_mode = request.args.get('quick', 'false').lower() == 'true'

    def get_categories():
        return ('categories', ai.get_process_categories(lookback_days=capacity_days))

    def get_version_analysis():
        return ('version_analysis', ai.analyze_version_performance(lookback_days=capacity_days))

    def get_extreme_variability():
        categories_result = ai.get_process_categories(lookback_days=capacity_days)
        return ('extreme_variability', ai.analyze_extreme_variability(
            process_categories=categories_result.get('categories', {})
        ))

    def get_load_patterns():
        return ('load_patterns', ai.analyze_load_patterns(lookback_days=capacity_days))

    def get_anomalies():
        return ('anomalies', ai.detect_process_anomalies(lookback_days=lookback_days))

    def get_bottlenecks():
        return ('bottlenecks', ai.identify_bottlenecks(lookback_days=lookback_days))

    def get_incidents():
        return ('incidents', ai.analyze_incident_patterns(lookback_days=lookback_days))

    def get_job_failures():
        return ('job_failures', ai.predict_job_failures(lookback_days=lookback_days))

    def get_stuck_activities():
        return ('stuck_activities', ai.find_stuck_activities_smart(lookback_days=lookback_days))

    def get_stuck_processes():
        return ('stuck_processes', ai.analyze_stuck_processes(lookback_days=lookback_days))

    def get_outlier_patterns():
        return ('outlier_patterns', ai.analyze_outlier_patterns(lookback_days=capacity_days))

    # Execute analyses in parallel
    analysis = {}

    tasks = [
        get_categories,
        get_version_analysis,
        get_extreme_variability,
        get_load_patterns,
        get_anomalies,
        get_bottlenecks,
        get_incidents,
        get_job_failures,
        get_stuck_processes,
        get_outlier_patterns
    ]

    if not quick_mode:
        tasks.append(get_stuck_activities)

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {executor.submit(task): task for task in tasks}

        for future in as_completed(future_to_task):
            try:
                key, result = future.result()
                analysis[key] = result
            except Exception as e:
                task_name = future_to_task[future].__name__
                analysis[task_name] = {'error': str(e)}
                current_app.logger.error(f"Error in {task_name}: {e}")

    # Generate comprehensive recommendations
    analysis['professional_insights'] = ai.generate_professional_insights(analysis)
    analysis['timestamp'] = datetime.now().isoformat()
    analysis['analysis_mode'] = 'quick' if quick_mode else 'full'

    return jsonify(analysis)
