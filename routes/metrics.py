"""
Metrics routes - Prometheus and health checks
"""
from flask import Blueprint, jsonify
import requests
from helpers.error_handler import safe_execute
from helpers.db_helper import get_db_helper
from services.camunda_service import collect_engine_health

metrics_bp = Blueprint('metrics', __name__)


@metrics_bp.route('/health')
def health_check():
    """Health check endpoint for the monitor itself"""
    from datetime import datetime
    from config import Config

    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }

    overall_healthy = True
    camunda_nodes = Config.load_camunda_nodes()
    camunda_auth = Config.get_camunda_auth()
    ssl_verify = Config.SSL_VERIFY

    # Check database connectivity
    try:
        db_helper = get_db_helper()
        success, latency = db_helper.test_connection()
        pool_stats = db_helper.get_pool_stats()
        health_status['checks']['database'] = {
            'status': 'up' if success else 'down',
            'latency_ms': latency,
            'pool_status': pool_stats.get('status', 'unknown')
        }
        if not success:
            overall_healthy = False
    except RuntimeError:
        health_status['checks']['database'] = {
            'status': 'down',
            'error': 'Database helper not initialized'
        }
        overall_healthy = False
    except Exception as e:
        health_status['checks']['database'] = {
            'status': 'down',
            'error': str(e)
        }
        overall_healthy = False

    # Check Camunda node connectivity
    reachable_nodes = 0
    total_nodes = len(camunda_nodes)
    for name, url in camunda_nodes.items():
        try:
            response = requests.get(f"{url}/engine", auth=camunda_auth, timeout=5, verify=ssl_verify)
            if response.status_code == 200:
                reachable_nodes += 1
        except:
            pass

    health_status['checks']['camunda_nodes'] = {
        'reachable': reachable_nodes,
        'total': total_nodes,
        'status': 'up' if reachable_nodes > 0 else 'down'
    }

    if reachable_nodes == 0:
        overall_healthy = False

    # Check circuit breaker states
    from services.camunda_service import api_circuit_breaker, jmx_circuit_breaker
    health_status['checks']['circuit_breakers'] = {
        'api': api_circuit_breaker.get_state(),
        'jmx': jmx_circuit_breaker.get_state()
    }

    if overall_healthy:
        return jsonify(health_status), 200
    else:
        health_status['status'] = 'unhealthy'
        return jsonify(health_status), 503


@metrics_bp.route('/metrics')
def prometheus_metrics():
    """Export metrics in Prometheus format"""
    # Use safe_execute to prevent crashes
    data = safe_execute(
        lambda: collect_engine_health(),
        default_value={
            'cluster_status': {},
            'totals': {},
            'cluster_nodes': [],
            'db_metrics': {}
        },
        context="Collecting metrics for Prometheus"
    )

    if not data:
        return f"# ERROR: Failed to collect metrics\n", 500, {'Content-Type': 'text/plain; charset=utf-8'}

    lines = []
    lines.append("# HELP camunda_cluster_info Cluster information")
    lines.append("# TYPE camunda_cluster_info gauge")

    cluster = data.get('cluster_status', {})
    version = cluster.get("engine_version", "unknown")
    total_nodes = cluster.get("total_nodes", 0)
    lines.append(f'camunda_cluster_info{{version="{version}"}} {total_nodes}')

    lines.append("# HELP camunda_cluster_running_nodes Number of running nodes")
    lines.append("# TYPE camunda_cluster_running_nodes gauge")
    lines.append(f'camunda_cluster_running_nodes {cluster.get("running_nodes", 0)}')

    totals = data.get('totals', {})

    # Core metrics
    metrics_map = {
        'active_instances': 'Active process instances',
        'user_tasks': 'Active user tasks',
        'external_tasks': 'Active external tasks',
        'incidents': 'Active incidents',
        'total_jobs': 'Total jobs',
        'failed_jobs': 'Jobs with no retries left',
        'deployment_count': 'Total deployments',
        'process_definitions': 'Process definitions',
        'dmn_definitions': 'DMN definitions'
    }

    for metric_key, help_text in metrics_map.items():
        metric_name = f'camunda_{metric_key}'
        lines.append(f"# HELP {metric_name} {help_text}")
        lines.append(f"# TYPE {metric_name} gauge")
        lines.append(f'{metric_name} {totals.get(metric_key, 0)}')

    # Per-node metrics
    nodes = data.get('cluster_nodes', [])

    # Node status
    lines.append("# HELP camunda_node_status Node status (1=RUNNING, 0=ERROR/DOWN)")
    lines.append("# TYPE camunda_node_status gauge")
    for node in nodes:
        status = 1 if node.get('status') == 'RUNNING' else 0
        node_name = node.get('name', 'unknown').replace('-', '_')
        lines.append(f'camunda_node_status{{node="{node_name}",url="{node.get("url", "")}"}} {status}')

    # Node response time
    lines.append("# HELP camunda_node_response_time_ms Node response time in milliseconds")
    lines.append("# TYPE camunda_node_response_time_ms gauge")
    for node in nodes:
        if node.get('response_time_ms') is not None:
            node_name = node.get('name', 'unknown').replace('-', '_')
            lines.append(f'camunda_node_response_time_ms{{node="{node_name}"}} {node.get("response_time_ms", 0)}')

    # JVM metrics
    jvm_metrics = {
        'heap_used_mb': 'JVM heap used in MB',
        'heap_utilization_percent': 'JVM heap utilization percentage',
        'cpu_load_percent': 'CPU load percentage'
    }

    for metric_suffix, help_text in jvm_metrics.items():
        metric_name = f'camunda_jvm_{metric_suffix}'
        lines.append(f"# HELP {metric_name} {help_text}")
        lines.append(f"# TYPE {metric_name} gauge")

        for node in nodes:
            jvm = node.get('jvm_metrics', {})
            if jvm.get('status') == 'HEALTHY':
                node_name = node.get('name', 'unknown').replace('-', '_')

                if 'heap' in metric_suffix:
                    value = jvm.get("memory", {}).get(metric_suffix.replace('_percent', '_pct'), 0)
                else:
                    value = jvm.get("system", {}).get(metric_suffix.replace('_percent', '_pct'), 0)

                lines.append(f'{metric_name}{{node="{node_name}"}} {value}')

    # Database metrics
    db_metrics = data.get('db_metrics', {})
    db_metric_names = {
        'latency_ms': 'Database query latency in milliseconds',
        'active_connections': 'Active database connections',
        'connection_utilization': 'Connection pool utilization percentage'
    }

    for metric_key, help_text in db_metric_names.items():
        metric_name = f'camunda_db_{metric_key}'
        lines.append(f"# HELP {metric_name} {help_text}")
        lines.append(f"# TYPE {metric_name} gauge")
        lines.append(f'{metric_name} {db_metrics.get(metric_key, 0)}')

    ai_metrics_result = safe_execute(
        lambda: _collect_ai_metrics_for_prometheus(data),
        default_value={},
        context="Collecting AI metrics for Prometheus",
        log_errors=False
    )

    if ai_metrics_result:
        # Cluster health score
        if 'health_score' in ai_metrics_result:
            lines.append("# HELP camunda_ai_health_score AI-calculated cluster health score (0-100)")
            lines.append("# TYPE camunda_ai_health_score gauge")
            lines.append(f'camunda_ai_health_score {ai_metrics_result["health_score"]}')

        # Anomaly detection
        if 'anomaly_count' in ai_metrics_result:
            lines.append("# HELP camunda_ai_anomalies_detected Number of process anomalies detected")
            lines.append("# TYPE camunda_ai_anomalies_detected gauge")
            lines.append(f'camunda_ai_anomalies_detected {ai_metrics_result["anomaly_count"]}')

        if 'critical_anomalies' in ai_metrics_result:
            lines.append("# HELP camunda_ai_anomalies_critical Number of critical anomalies")
            lines.append("# TYPE camunda_ai_anomalies_critical gauge")
            lines.append(f'camunda_ai_anomalies_critical {ai_metrics_result["critical_anomalies"]}')

        # Node performance scores
        if 'node_scores' in ai_metrics_result:
            lines.append("# HELP camunda_ai_node_performance_score AI-calculated node performance score (0-100)")
            lines.append("# TYPE camunda_ai_node_performance_score gauge")
            for node_performance in ai_metrics_result['node_scores'].get('rankings', []):
                lines.append(f'camunda_ai_node_performance_score{{node="{node_performance['node_name']}"}} {node_performance['performance_score']}')

    return '\n'.join(lines) + '\n', 200, {'Content-Type': 'text/plain; charset=utf-8'}


def _collect_ai_metrics_for_prometheus(cluster_data):
    """Collect FAST AI metrics for Prometheus export using lightweight COUNT queries only

    Args:
        cluster_data: Already collected cluster health data from collect_engine_health()

    Note: This uses fast aggregation queries instead of heavy AI processing
          to keep Prometheus scraping fast (< 100ms added overhead)
    """
    from services.ai_service import get_ai_analytics
    from helpers.db_helper import execute_query
    from flask import current_app

    ai = get_ai_analytics()
    metrics = {}

    # Extract db_metrics from already collected data
    db_metrics = cluster_data.get('db_metrics', {})

    # 1. Health Score (FAST - uses existing cluster data, no DB queries)
    try:
        health_result = ai.get_cluster_health_score(cluster_data, db_metrics)
        metrics['health_score'] = health_result.get('overall_score', 0)
        metrics['node_scores'] = ai.analyze_node_performance(cluster_data.get('cluster_nodes', []))
    except:
        pass

    # 2. Anomaly Detection (FAST - simple count, not full analysis)
    try:
        lookback_days = int(current_app.config.get('AI_LOOKBACK_DAYS', 30))
        # Fast count: processes with very long running instances (> 1 hour avg)
        query = f"""
            SELECT COUNT(DISTINCT proc_def_key_) as anomaly_count
            FROM (
                SELECT 
                    proc_def_key_,
                    AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_))) as avg_duration_sec
                FROM act_hi_procinst
                WHERE end_time_ IS NOT NULL
                AND end_time_ > NOW() - INTERVAL '{lookback_days} days'
                GROUP BY proc_def_key_
                HAVING AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_))) > 3600
            ) slow_processes
        """
        result = execute_query(query)
        if result and len(result) > 0:
            metrics['anomaly_count'] = int(result[0].get('anomaly_count', 0))
            # For critical: > 6 hours average
            query_critical = query.replace('> 3600', '> 21600')
            result_critical = execute_query(query_critical)
            if result_critical and len(result_critical) > 0:
                metrics['critical_anomalies'] = int(result_critical[0].get('anomaly_count', 0))
    except:
        pass

    return metrics
