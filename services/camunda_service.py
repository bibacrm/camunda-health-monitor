"""
Camunda Service
Handles all Camunda-related operations, health checks, and metrics collection
"""
import time
import logging
import concurrent.futures
import requests
from datetime import datetime
from collections import defaultdict

from helpers.error_handler import CircuitBreaker
from helpers.db_helper import execute_query

logger = logging.getLogger('champa_monitor.camunda_service')

# Create circuit breakers for external services (initialized once)
api_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60, name="Camunda API")
jmx_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30, name="JMX")

# Configuration will be loaded lazily when needed
_config_cache = {}


def _get_config():
    """Lazy-load configuration to avoid circular imports"""
    if not _config_cache:
        from config import Config
        _config_cache['CAMUNDA_NODES'] = Config.load_camunda_nodes()
        _config_cache['JMX_ENDPOINTS'] = Config.load_jmx_endpoints()
        _config_cache['CAMUNDA_AUTH'] = Config.get_camunda_auth()
        _config_cache['SSL_VERIFY'] = Config.SSL_VERIFY
        _config_cache['JVM_METRICS_SOURCE'] = Config.JVM_METRICS_SOURCE
    return _config_cache


def collect_engine_health():
    """
    Collect comprehensive engine health metrics
    Main entry point for health data collection
    """
    logger.info("Collecting health metrics...")
    collection_start = time.time()

    # Get configuration
    config = _get_config()
    CAMUNDA_NODES = config['CAMUNDA_NODES']
    JMX_ENDPOINTS = config['JMX_ENDPOINTS']

    # Initialize metrics
    cluster_status = {
        "total_nodes": len(CAMUNDA_NODES),
        "running_nodes": 0,
        "engine_version": None,
        "issues": []
    }

    totals = {
        "active_instances": 0,
        "user_tasks": 0,
        "external_tasks": 0,
        "incidents": 0,
        "total_jobs": 0,
        "failed_jobs": 0,
        "active_jobs": 0,
        "deployment_count": 0,
        "process_definitions": 0,
        "dmn_definitions": 0,
        "pending_messages": 0,
        "pending_signals": 0,
        "stuck_instances": 0,
        "jobs_executed_per_min": 0,
        "jobs_executed_total": 0
    }

    # Collect JMX metrics and node data in parallel
    jmx_data = {}
    max_workers = len(CAMUNDA_NODES) + len(JMX_ENDPOINTS) if JMX_ENDPOINTS else len(CAMUNDA_NODES)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit JMX collection tasks if endpoints configured
        if JMX_ENDPOINTS:
            raw_jmx_future = executor.submit(_collect_jmx_metrics)

        # Submit node data collection tasks
        node_futures = [
            executor.submit(_fetch_node_data, name, url, {})
            for name, url in CAMUNDA_NODES.items()
        ]

        # Wait for JMX metrics first
        if JMX_ENDPOINTS:
            raw_jmx = raw_jmx_future.result()

            # Extract JVM health metrics in parallel
            extract_futures = {}
            JVM_METRICS_SOURCE = config['JVM_METRICS_SOURCE']
            for name, raw_metrics in raw_jmx.items():
                if JVM_METRICS_SOURCE == 'micrometer':
                    extract_futures[name] = executor.submit(_extract_jvm_health_metrics_quarkus, raw_metrics)
                else:
                    extract_futures[name] = executor.submit(_extract_jvm_health_metrics, raw_metrics)

            # Collect extracted JVM metrics
            for name, future in extract_futures.items():
                jmx_data[name] = future.result()

        # Collect node metrics
        cluster_metrics = [future.result() for future in concurrent.futures.as_completed(node_futures)]

        # Update node metrics with JMX data if available
        for node in cluster_metrics:
            if node['name'] in jmx_data:
                node['jvm_metrics'] = jmx_data[node['name']]
                node['jvm_status'] = jmx_data[node['name']].get('status', 'UNKNOWN')

    # Sort by node name
    cluster_metrics.sort(key=lambda x: x['name'])

    # Count running nodes and aggregate totals
    for node in cluster_metrics:
        if node["status"] == "RUNNING":
            cluster_status["running_nodes"] += 1
            if not cluster_status["engine_version"]:
                cluster_status["engine_version"] = "7.x"
            totals["jobs_executed_total"] += node["jobs_successful"]
        else:
            cluster_status["issues"].append(
                f"Node {node['url']}: {node.get('error', node['status'])}"
            )

    # Collect database health check and API calls in parallel
    db_metrics = {"connectivity": "OK"}
    first_node = next((n for n in cluster_metrics if n["status"] == "RUNNING"), None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        # Database health check
        db_start = time.time()
        db_ping_future = executor.submit(execute_query, "SELECT 1")
        db_conn_stats_future = executor.submit(
            execute_query,
            "SELECT count(*) AS active, "
            "(SELECT setting::int FROM pg_settings WHERE name='max_connections') AS max "
            "FROM pg_stat_activity WHERE datname=current_database();"
        )

        # API calls to first running node
        api_calls = {}
        if first_node:
            api_calls = _submit_api_calls(executor, first_node)

        # Process database results
        try:
            db_ping_future.result()
            conn_stats = db_conn_stats_future.result()

            db_metrics["latency_ms"] = int((time.time() - db_start) * 1000)

            if conn_stats:
                db_metrics.update({
                    "active_connections": conn_stats[0]["active"],
                    "max_connections": conn_stats[0]["max"],
                    "connection_utilization": round(
                        (conn_stats[0]["active"] / conn_stats[0]["max"] * 100), 1
                    ) if conn_stats[0]["max"] > 0 else 0
                })
        except Exception as e:
            db_metrics.update({
                "connectivity": "ERROR",
                "error": str(e)
            })
            cluster_status["issues"].append(f"Database: {str(e)}")

        # Process API call results
        _process_api_results(api_calls, totals)

    # Set shared data for all running nodes
    for node in cluster_metrics:
        if node["status"] == "RUNNING":
            node.update({
                "active_instances": totals["active_instances"],
                "user_tasks": totals["user_tasks"],
                "external_tasks": totals["external_tasks"],
                "incidents": totals["incidents"],
                "deployment_count": totals["deployment_count"],
                "process_definitions": totals["process_definitions"],
                "dmn_definitions": totals["dmn_definitions"]
            })

    # Calculate jobs per minute
    if totals["jobs_executed_total"] > 0:
        totals["jobs_executed_per_min"] = round(totals["jobs_executed_total"] / len(CAMUNDA_NODES), 1)

    collection_duration = int((time.time() - collection_start) * 1000)
    logger.info(f"Health collection complete in {collection_duration}ms: "
                f"{cluster_status['running_nodes']}/{cluster_status['total_nodes']} nodes, "
                f"{totals['active_instances']} instances, "
                f"{totals['incidents']} incidents")

    return {
        "cluster_nodes": cluster_metrics,
        "cluster_status": cluster_status,
        "totals": totals,
        "db_metrics": db_metrics,
        "timestamp": datetime.now()
    }


# ============================================================
# Helper Functions
# ============================================================

def _collect_jmx_metrics():
    """Collect JVM metrics from JMX exporters"""
    config = _get_config()
    JMX_ENDPOINTS = config['JMX_ENDPOINTS']
    CAMUNDA_AUTH = config['CAMUNDA_AUTH']
    SSL_VERIFY = config['SSL_VERIFY']

    jmx_metrics = {}

    def fetch_jmx(name, url):
        try:
            response = jmx_circuit_breaker.call(
                requests.get, url, auth=CAMUNDA_AUTH, timeout=10, verify=SSL_VERIFY
            )
            if response.status_code == 200:
                return name, _parse_prometheus_metrics(response.text)
            return name, {'error': f'HTTP {response.status_code}'}
        except Exception as e:
            logger.warning(f"JMX collection failed for {name}: {e}")
            return name, {'error': str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(JMX_ENDPOINTS)) as executor:
        futures = [executor.submit(fetch_jmx, name, url) for name, url in JMX_ENDPOINTS.items()]
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            jmx_metrics[name] = result

    return jmx_metrics


def _parse_prometheus_metrics(metrics_text):
    """Parse Prometheus metrics format into a dictionary"""
    metrics = {}
    for line in metrics_text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        try:
            if ' ' in line:
                metric_part, value = line.rsplit(' ', 1)
                if '{' in metric_part:
                    metric_name = metric_part.split('{')[0]
                    labels_part = metric_part.split('{')[1].rstrip('}')
                    labels = {}
                    for label in labels_part.split(','):
                        if '=' in label:
                            key, val = label.split('=', 1)
                            labels[key.strip()] = val.strip().strip('"')
                    if metric_name not in metrics:
                        metrics[metric_name] = {}
                    label_key = '_'.join([f"{k}_{v}" for k, v in labels.items()]) if labels else 'default'
                    metrics[metric_name][label_key] = float(value)
                else:
                    metrics[metric_part] = float(value)
        except (ValueError, IndexError):
            continue
    return metrics


def _fetch_node_data(node_name, node_url, jmx_data):
    """Fetch comprehensive health data from a single Camunda node"""
    config = _get_config()
    CAMUNDA_AUTH = config['CAMUNDA_AUTH']
    SSL_VERIFY = config['SSL_VERIFY']

    node_url = node_url.strip()
    node_metrics = {
        "url": node_url,
        "name": node_name,
        "status": "DOWN",
        "response_time_ms": None,
        "error": None,
        "jvm_metrics": jmx_data.get(node_name, {'status': 'NO_JMX_DATA'}),
        "external_tasks_locked": 0,
        "external_tasks_no_retries": 0,
        "incident_types": {},
        "job_acquisition_attempts": 0,
        "job_acquired_success": 0,
        "job_acquired_failure": 0,
        "job_execution_rejected": 0,
        "jobs_successful": 0,
        "jobs_failed": 0,
        "job_acquisition_success_rate": 100,
        "job_success_rate": 100,
        "decision_instances": 0,
        "flow_node_instances": 0,
        "process_instances_started": 0,
        "workload_score": 0,
        "jvm_status": "UNKNOWN"
    }

    try:
        start_time = time.time()
        response = requests.get(f"{node_url}/engine", auth=CAMUNDA_AUTH, timeout=10, verify=SSL_VERIFY)
        response.raise_for_status()
        engines = response.json()

        node_metrics["response_time_ms"] = int((time.time() - start_time) * 1000)

        if not engines:
            node_metrics["status"] = "NO_ENGINES"
            node_metrics["error"] = "Node is running but reports no active engines."
            return node_metrics

        node_metrics["status"] = "RUNNING"
        node_metrics["jvm_status"] = node_metrics["jvm_metrics"].get('status', 'UNKNOWN')

        # Collect additional metrics in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            ext_tasks_active_future = executor.submit(
                requests.get, f"{node_url}/external-task/count?locked=true",
                auth=CAMUNDA_AUTH, timeout=10, verify=SSL_VERIFY
            )
            ext_tasks_failed_future = executor.submit(
                requests.get, f"{node_url}/external-task/count?noRetriesLeft=true",
                auth=CAMUNDA_AUTH, timeout=10, verify=SSL_VERIFY
            )
            incidents_future = executor.submit(
                requests.get, f"{node_url}/incident/count",
                auth=CAMUNDA_AUTH, timeout=10, verify=SSL_VERIFY
            )
            metrics_future = executor.submit(
                requests.get, f"{node_url}/engine/default/metrics",
                auth=CAMUNDA_AUTH, timeout=10, verify=SSL_VERIFY
            )

            # Process external tasks
            try:
                ext_resp = ext_tasks_active_future.result()
                if ext_resp.status_code == 200:
                    node_metrics["external_tasks_locked"] = ext_resp.json().get('count', 0)
            except:
                pass

            try:
                ext_failed_resp = ext_tasks_failed_future.result()
                if ext_failed_resp.status_code == 200:
                    node_metrics["external_tasks_no_retries"] = ext_failed_resp.json().get('count', 0)
            except:
                pass

            # Process incidents
            try:
                incidents_resp = incidents_future.result()
                if incidents_resp.status_code == 200:
                    incidents = incidents_resp.json()
                    incident_count = incidents.get("count", 0)
                    node_metrics["incident_types"] = {"ALL": incident_count}
            except:
                pass

            # Process metrics with rate calculation
            try:
                metrics_resp = metrics_future.result()
                if metrics_resp.status_code == 200:
                    metrics_data = metrics_resp.json()

                    # Group metrics by name
                    metrics_by_name = defaultdict(list)
                    for metric in metrics_data:
                        if metric.get("timestamp") and metric.get("name"):
                            try:
                                timestamp_str = metric["timestamp"]
                                if timestamp_str.endswith('Z'):
                                    timestamp_str = timestamp_str[:-1] + '+00:00'
                                elif timestamp_str.endswith('+0000'):
                                    timestamp_str = timestamp_str[:-5] + '+00:00'
                                metric_time = datetime.fromisoformat(timestamp_str)
                                metrics_by_name[metric["name"]].append({
                                    "value": metric.get("value", 0),
                                    "timestamp": metric_time
                                })
                            except (ValueError, TypeError):
                                continue

                    # Calculate rates
                    node_rates = {}
                    for metric_name, values in metrics_by_name.items():
                        sorted_values = sorted(values, key=lambda x: x["timestamp"], reverse=True)
                        if len(sorted_values) >= 2:
                            latest, previous = sorted_values[0], sorted_values[1]
                            time_diff = (latest["timestamp"] - previous["timestamp"]).total_seconds() / 60
                            if time_diff > 0:
                                node_rates[metric_name] = max(0, latest["value"] - previous["value"])
                        elif len(sorted_values) == 1:
                            node_rates[metric_name] = sorted_values[0]["value"]

                    # Map metrics
                    node_metrics["job_acquisition_attempts"] = int(node_rates.get("job-acquisition-attempt", 0))
                    node_metrics["job_acquired_success"] = int(node_rates.get("job-acquired-success", 0))
                    node_metrics["job_acquired_failure"] = int(node_rates.get("job-acquired-failure", 0))
                    node_metrics["job_execution_rejected"] = int(node_rates.get("job-execution-rejected", 0))
                    node_metrics["jobs_successful"] = int(node_rates.get("job-successful", 0))
                    node_metrics["jobs_failed"] = int(node_rates.get("job-failed", 0))
                    node_metrics["decision_instances"] = int(node_rates.get("decision-instances", 0))
                    node_metrics["flow_node_instances"] = int(node_rates.get("flow-node-instances", 0))
                    node_metrics["process_instances_started"] = int(node_rates.get("process-instances", 0))

                    # Calculate success rates
                    attempts = node_metrics["job_acquisition_attempts"]
                    success = node_metrics["job_acquired_success"]
                    node_metrics["job_acquisition_success_rate"] = round(
                        (success / attempts) * 100, 1
                    ) if attempts > 0 else 100

                    total_jobs = node_metrics["jobs_successful"] + node_metrics["jobs_failed"]
                    node_metrics["job_success_rate"] = round(
                        (node_metrics["jobs_successful"] / total_jobs) * 100, 1
                    ) if total_jobs > 0 else 100

                    # Calculate workload score
                    node_metrics["workload_score"] = (
                        attempts +
                        node_metrics["decision_instances"] +
                        node_metrics["flow_node_instances"]
                    )
            except Exception as e:
                logger.error(f"Metrics processing error for {node_url}: {e}", exc_info=True)

    except Exception as e:
        node_metrics["status"] = "ERROR"
        node_metrics["error"] = str(e)

    return node_metrics


def _submit_api_calls(executor, first_node):
    """Submit API calls to Camunda REST API"""
    config = _get_config()
    CAMUNDA_AUTH = config['CAMUNDA_AUTH']
    SSL_VERIFY = config['SSL_VERIFY']

    return {
        'instances': executor.submit(
            requests.get, f"{first_node['url']}/process-instance/count",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        ),
        'tasks': executor.submit(
            requests.get, f"{first_node['url']}/task/count",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        ),
        'ext_tasks': executor.submit(
            requests.get, f"{first_node['url']}/external-task/count",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        ),
        'incidents': executor.submit(
            requests.get, f"{first_node['url']}/incident/count",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        ),
        'deployments': executor.submit(
            requests.get, f"{first_node['url']}/deployment/count",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        ),
        'process_defs': executor.submit(
            requests.get, f"{first_node['url']}/process-definition/count",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        ),
        'dmn_defs': executor.submit(
            requests.get, f"{first_node['url']}/decision-definition/count",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        ),
        'jobs_total': executor.submit(
            requests.get, f"{first_node['url']}/job/count",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        ),
        'jobs_active': executor.submit(
            requests.get, f"{first_node['url']}/job/count?executable=true",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        ),
        'jobs_failed': executor.submit(
            requests.get, f"{first_node['url']}/job/count?noRetriesLeft=true",
            timeout=10, auth=CAMUNDA_AUTH, verify=SSL_VERIFY
        )
    }


def _process_api_results(api_calls, totals):
    """Process API call results and update totals"""
    for key, future in api_calls.items():
        try:
            resp = future.result()
            if resp.status_code == 200:
                count = resp.json().get('count', 0)
                if key == 'instances':
                    totals['active_instances'] = count
                elif key == 'tasks':
                    totals['user_tasks'] = count
                elif key == 'ext_tasks':
                    totals['external_tasks'] = count
                elif key == 'incidents':
                    totals['incidents'] = count
                elif key == 'deployments':
                    totals['deployment_count'] = count
                elif key == 'process_defs':
                    totals['process_definitions'] = count
                elif key == 'dmn_defs':
                    totals['dmn_definitions'] = count
                elif key == 'jobs_total':
                    totals['total_jobs'] = count
                elif key == 'jobs_active':
                    totals['active_jobs'] = count
                elif key == 'jobs_failed':
                    totals['failed_jobs'] = count
        except Exception as e:
            logger.error(f"API call {key} failed: {e}", exc_info=True)


def _extract_jvm_health_metrics_quarkus(quarkus_data):
    """Extract key JVM health metrics from Quarkus/Micrometer format"""
    if 'error' in quarkus_data:
        return {'status': 'ERROR', 'error': quarkus_data['error']}

    try:
        # Memory metrics
        heap_used = 0
        heap_max = 0
        heap_committed = 0
        nonheap_used = 0

        memory_used = quarkus_data.get('jvm_memory_used_bytes', {})
        for key, value in memory_used.items():
            if key.startswith('area_heap'):
                heap_used += value
            elif key.startswith('area_nonheap'):
                nonheap_used += value

        memory_max = quarkus_data.get('jvm_memory_max_bytes', {})
        for key, value in memory_max.items():
            if key.startswith('area_heap') and value > 0:
                heap_max += value

        if heap_used == 0:
            for key, value in memory_used.items():
                if 'G1 Eden Space' in key or 'G1 Old Gen' in key or 'G1 Survivor Space' in key:
                    heap_used += value

        if heap_max == 0:
            for key, value in memory_max.items():
                if 'G1 Old Gen' in key and value > 0:
                    heap_max = value
                    break
            if heap_max == 0:
                heap_max = quarkus_data.get('jvm_gc_max_data_size_bytes', 1)

        memory_committed = quarkus_data.get('jvm_memory_committed_bytes', {})
        for key, value in memory_committed.items():
            if key.startswith('area_heap') or 'G1' in key:
                heap_committed += value

        # GC metrics
        gc_pause_data = quarkus_data.get('jvm_gc_pause_seconds_count', {})
        gc_pause_time = quarkus_data.get('jvm_gc_pause_seconds_sum', {})

        minor_collections = 0
        minor_time = 0
        major_collections = 0
        major_time = 0

        for key, count in gc_pause_data.items():
            time_value = gc_pause_time.get(key, 0)
            if 'minor GC' in key or 'end of minor GC' in key:
                minor_collections += count
                minor_time += time_value
            elif 'major GC' in key or 'end of major GC' in key:
                major_collections += count
                major_time += time_value
            else:
                minor_collections += count
                minor_time += time_value

        # Thread metrics
        thread_count = quarkus_data.get('jvm_threads_live_threads', 32)
        thread_peak = quarkus_data.get('jvm_threads_peak_threads', 32)
        daemon_threads = quarkus_data.get('jvm_threads_daemon_threads', 26)

        # System metrics
        cpu_load = quarkus_data.get('process_cpu_usage', 0)
        system_load = quarkus_data.get('system_load_average_1m', 0)
        system_cpu = quarkus_data.get('system_cpu_usage', 0)

        # File descriptors
        open_fds = quarkus_data.get('process_files_open_files', 0)
        max_fds = quarkus_data.get('process_files_max_files', 1)

        return {
            'status': 'HEALTHY',
            'memory': {
                'heap_used_mb': heap_used / 1024 / 1024,
                'heap_max_mb': heap_max / 1024 / 1024,
                'heap_utilization_pct': (heap_used / heap_max * 100) if heap_max > 0 else 0,
                'heap_committed_mb': heap_committed / 1024 / 1024,
                'nonheap_used_mb': nonheap_used / 1024 / 1024
            },
            'gc': {
                'minor_collections': minor_collections,
                'minor_time_sec': minor_time,
                'major_collections': major_collections,
                'major_time_sec': major_time,
                'total_gc_time_sec': minor_time + major_time
            },
            'threads': {
                'current': thread_count,
                'peak': thread_peak,
                'daemon': daemon_threads
            },
            'system': {
                'cpu_load_pct': cpu_load * 100,
                'system_load': system_load,
                'memory_free_mb': 0,
                'memory_total_mb': 1,
                'memory_utilization_pct': system_cpu * 100
            },
            'file_descriptors': {
                'open': open_fds,
                'max': max_fds,
                'utilization_pct': (open_fds / max_fds * 100) if max_fds > 0 else 0
            }
        }
    except Exception as e:
        return {'status': 'PARSE_ERROR', 'error': str(e)}


def _extract_jvm_health_metrics(jmx_data):
    """Extract key JVM health metrics from standard JMX exporter format"""
    if 'error' in jmx_data:
        return {'status': 'ERROR', 'error': jmx_data['error']}

    try:
        # Memory metrics - support both old (0.20.0) and new (1.0.1+) naming
        memory_used = jmx_data.get('jvm_memory_used_bytes', jmx_data.get('jvm_memory_bytes_used', {}))
        memory_max = jmx_data.get('jvm_memory_max_bytes', jmx_data.get('jvm_memory_bytes_max', {}))
        memory_committed = jmx_data.get('jvm_memory_committed_bytes', jmx_data.get('jvm_memory_bytes_committed', {}))

        heap_used = memory_used.get('area_heap', 0)
        heap_max = memory_max.get('area_heap', 1)
        heap_committed = memory_committed.get('area_heap', 0)
        nonheap_used = memory_used.get('area_nonheap', 0)

        # GC metrics
        gc_copy_count = jmx_data.get('jvm_gc_collection_seconds_count', {}).get('gc_Copy', 0)
        gc_copy_time = jmx_data.get('jvm_gc_collection_seconds_sum', {}).get('gc_Copy', 0)
        gc_mark_count = jmx_data.get('jvm_gc_collection_seconds_count', {}).get('gc_MarkSweepCompact', 0)
        gc_mark_time = jmx_data.get('jvm_gc_collection_seconds_sum', {}).get('gc_MarkSweepCompact', 0)

        # Thread metrics
        thread_count = jmx_data.get('jvm_threads_current', 32)
        thread_peak = jmx_data.get('jvm_threads_peak', 32)
        daemon_threads = jmx_data.get('jvm_threads_daemon', 26)

        # System metrics
        cpu_load = jmx_data.get('jvm_os_processcpuload', 0)
        system_load = jmx_data.get('jvm_os_systemloadaverage', 0)
        free_memory = jmx_data.get('jvm_os_freememorysize', 0)
        total_memory = jmx_data.get('jvm_os_totalmemorysize', 1)

        # File descriptors
        open_fds = jmx_data.get('jvm_os_openfiledescriptorcount', 0)
        max_fds = jmx_data.get('jvm_os_maxfiledescriptorcount', 1)

        return {
            'status': 'HEALTHY',
            'memory': {
                'heap_used_mb': heap_used / 1024 / 1024,
                'heap_max_mb': heap_max / 1024 / 1024,
                'heap_utilization_pct': (heap_used / heap_max * 100) if heap_max > 0 else 0,
                'heap_committed_mb': heap_committed / 1024 / 1024,
                'nonheap_used_mb': nonheap_used / 1024 / 1024
            },
            'gc': {
                'minor_collections': gc_copy_count,
                'minor_time_sec': gc_copy_time,
                'major_collections': gc_mark_count,
                'major_time_sec': gc_mark_time,
                'total_gc_time_sec': gc_copy_time + gc_mark_time
            },
            'threads': {
                'current': thread_count,
                'peak': thread_peak,
                'daemon': daemon_threads
            },
            'system': {
                'cpu_load_pct': cpu_load * 100,
                'system_load': system_load,
                'memory_free_mb': free_memory / 1024 / 1024,
                'memory_total_mb': total_memory / 1024 / 1024,
                'memory_utilization_pct': (
                    ((total_memory - free_memory) / total_memory * 100)
                    if total_memory > 0 else 0
                )
            },
            'file_descriptors': {
                'open': open_fds,
                'max': max_fds,
                'utilization_pct': (open_fds / max_fds * 100) if max_fds > 0 else 0
            }
        }
    except Exception as e:
        return {'status': 'PARSE_ERROR', 'error': str(e)}


