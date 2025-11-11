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

    return '\n'.join(lines) + '\n', 200, {'Content-Type': 'text/plain; charset=utf-8'}

