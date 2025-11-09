"""
Camunda 7 Health Monitor
A lightweight monitoring dashboard for Camunda BPM Platform clusters

Copyright (c) 2025 Champa Intelligence (https://champa-bpmn.com)
Licensed for educational, non-profit, and personal use only.
Commercial use requires a separate license - contact info@champa-bpmn.com
"""
import os
import time
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from collections import defaultdict
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# ============================================================
# Logging Configuration
# ============================================================

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler with rotation (50MB max, keep 5 backups)
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=50*1024*1024,  # 50MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Console handler (optional - keep prints visible during development)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Configure app logger
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(logging.INFO)

# Create a general logger for non-Flask functions
logger = logging.getLogger('champa_monitor')
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# ============================================================
# Configuration from Environment Variables
# ============================================================

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'camunda'),
    'user': os.getenv('DB_USER', 'camunda'),
    'password': os.getenv('DB_PASSWORD', 'camunda'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# Load Camunda nodes from environment
CAMUNDA_NODES = {}
JMX_ENDPOINTS = {}
node_index = 1

while True:
    node_name = os.getenv(f'CAMUNDA_NODE_{node_index}_NAME')
    node_url = os.getenv(f'CAMUNDA_NODE_{node_index}_URL')
    if not node_name or not node_url:
        break
    CAMUNDA_NODES[node_name] = node_url

    # Optional JMX endpoint
    jmx_url = os.getenv(f'JMX_NODE_{node_index}_URL')
    if jmx_url:
        JMX_ENDPOINTS[node_name] = jmx_url

    node_index += 1

# If no nodes configured, use defaults
if not CAMUNDA_NODES:
    CAMUNDA_NODES = {
        'node1': os.getenv('CAMUNDA_URL', 'http://localhost:8080/engine-rest')
    }

CAMUNDA_AUTH = None
camunda_user = os.getenv('CAMUNDA_API_USER')
camunda_pass = os.getenv('CAMUNDA_API_PASSWORD')
if camunda_user and camunda_pass:
    CAMUNDA_AUTH = (camunda_user, camunda_pass)

JVM_METRICS_SOURCE = os.getenv('JVM_METRICS_SOURCE', 'jmx')
STUCK_INSTANCE_DAYS = int(os.getenv('STUCK_INSTANCE_DAYS', '7'))

logger.info("=" * 60)
logger.info("Camunda Health Monitor Starting")
logger.info("=" * 60)
logger.info(f"Monitoring {len(CAMUNDA_NODES)} node(s):")

for name, url in CAMUNDA_NODES.items():
    logger.info(f"  - {name}: {url}")
logger.info(f"Database: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
logger.info(f"JVM Metrics Source: {JVM_METRICS_SOURCE}")
logger.info(f"JMX Endpoints: {len(JMX_ENDPOINTS)} configured")
logger.info("=" * 60)


# ============================================================
# Database Connection
# ============================================================

def get_db_connection():
    """Get a database connection"""
    return psycopg2.connect(**DB_CONFIG)


def execute_query(query, params=None):
    """Execute a database query and return results"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Execute with or without params
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        # Check if this is a SELECT query
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            # Convert RealDictRow to regular dict explicitly
            return [dict(row) for row in results] if results else []
        else:
            conn.commit()
            return []
    except Exception as e:
        logger.error(f"Database query error: {e}", exc_info=True)
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ============================================================
# JMX Metrics Collection
# ============================================================

def parse_prometheus_metrics(metrics_text):
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


def collect_jmx_metrics():
    """Collect JVM metrics from JMX exporters"""
    jmx_metrics = {}

    def fetch_jmx(name, url):
        try:
            response = requests.get(url, auth=CAMUNDA_AUTH, timeout=10)
            if response.status_code == 200:
                return name, parse_prometheus_metrics(response.text)
            return name, {'error': f'HTTP {response.status_code}'}
        except Exception as e:
            return name, {'error': str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(JMX_ENDPOINTS)) as executor:
        futures = [executor.submit(fetch_jmx, name, url) for name, url in JMX_ENDPOINTS.items()]
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            jmx_metrics[name] = result

    return jmx_metrics


def extract_jvm_health_metrics_quarkus(quarkus_data):
    """Extract key JVM health metrics from Quarkus/Micrometer format"""
    if 'error' in quarkus_data:
        return {'status': 'ERROR', 'error': quarkus_data['error']}
    try:
        # Memory metrics - Quarkus uses different structure
        heap_used = 0
        heap_max = 0
        heap_committed = 0
        nonheap_used = 0

        # Process jvm_memory_used_bytes
        memory_used = quarkus_data.get('jvm_memory_used_bytes', {})
        for key, value in memory_used.items():
            if key.startswith('area_heap'):
                heap_used += value
            elif key.startswith('area_nonheap'):
                nonheap_used += value

        # Process jvm_memory_max_bytes
        memory_max = quarkus_data.get('jvm_memory_max_bytes', {})
        for key, value in memory_max.items():
            if key.startswith('area_heap') and value > 0:
                heap_max += value

        # If no aggregated heap data, use individual G1 regions
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

        # Process jvm_memory_committed_bytes
        memory_committed = quarkus_data.get('jvm_memory_committed_bytes', {})
        for key, value in memory_committed.items():
            if key.startswith('area_heap') or 'G1' in key:
                heap_committed += value

        # GC metrics - Quarkus uses jvm_gc_pause_seconds
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


def extract_jvm_health_metrics(jmx_data):
    """Extract key JVM health metrics from standard JMX exporter format"""
    if 'error' in jmx_data:
        return {'status': 'ERROR', 'error': jmx_data['error']}
    try:
        # Memory metrics
        heap_used = jmx_data.get('jvm_memory_bytes_used', {}).get('area_heap', 0)
        heap_max = jmx_data.get('jvm_memory_bytes_max', {}).get('area_heap', 1)
        heap_committed = jmx_data.get('jvm_memory_bytes_committed', {}).get('area_heap', 0)
        nonheap_used = jmx_data.get('jvm_memory_bytes_used', {}).get('area_nonheap', 0)

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
                'memory_utilization_pct': ((total_memory - free_memory) / total_memory * 100) if total_memory > 0 else 0
            },
            'file_descriptors': {
                'open': open_fds,
                'max': max_fds,
                'utilization_pct': (open_fds / max_fds * 100) if max_fds > 0 else 0
            }
        }
    except Exception as e:
        return {'status': 'PARSE_ERROR', 'error': str(e)}


# ============================================================
# Node Data Collection
# ============================================================

def fetch_node_data(node_name, node_url, jmx_data):
    """Fetch comprehensive health data from a single Camunda node"""
    node_url = node_url.strip()
    # Initialize with ALL metrics
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
        response = requests.get(f"{node_url}/engine", auth=CAMUNDA_AUTH, timeout=10)
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
                auth=CAMUNDA_AUTH, timeout=10
            )
            ext_tasks_failed_future = executor.submit(
                requests.get, f"{node_url}/external-task/count?noRetriesLeft=true",
                auth=CAMUNDA_AUTH, timeout=10
            )
            incidents_future = executor.submit(
                requests.get, f"{node_url}/incident/count",
                auth=CAMUNDA_AUTH, timeout=10
            )
            metrics_future = executor.submit(
                requests.get, f"{node_url}/engine/default/metrics",
                auth=CAMUNDA_AUTH, timeout=10
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

                    # Group metrics by name and calculate rates
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

                    # Calculate rates from latest vs previous readings
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

                    # Map metrics to node_metrics
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


# ============================================================
# Database Analytics
# ============================================================

def collect_database_metrics():
    """Collect database performance and storage metrics"""
    try:
        # Table sizes
        table_sizes = execute_query("""
            SELECT relname AS table_name, pg_total_relation_size(relid) AS size_bytes
            FROM pg_catalog.pg_statio_user_tables
            WHERE relname LIKE 'act_h%' OR relname LIKE 'act_r%'
            ORDER BY size_bytes DESC
            LIMIT 10
        """)
        # Archivable instances
        archivable = execute_query("""
            SELECT count(*) AS count
            FROM act_hi_procinst
            WHERE end_time_ IS NOT NULL
            AND end_time_ < now() - interval '90 days'
        """)

        # Slow queries (requires pg_stat_statements extension)
        slow_queries = []
        try:
            slow_queries = execute_query("""
                SELECT calls, mean_exec_time AS avg_ms, max_exec_time AS max_ms,
                       query as full_query, left(query, 120) AS query_preview
                FROM pg_stat_statements
                WHERE query LIKE '%act_%'
                ORDER BY mean_exec_time DESC
                LIMIT 10
            """)
        except:
            logger.warning(f"Slow queries error: {e}")
            pass  # pg_stat_statements not enabled

        return {
            "table_sizes": table_sizes,
            "archivable_instances": archivable[0]['count'] if archivable else 0,
            "slow_queries": slow_queries
        }
    except Exception as e:
        logger.error(f"Error collecting database metrics: {e}", exc_info=True)
        return {
            "table_sizes": [],
            "archivable_instances": 0,
            "slow_queries": []
        }


# ============================================================
# Main Health Collection
# ============================================================

def collect_engine_health():
    """Collect comprehensive engine health metrics"""
    logger.info(f"Collecting health metrics...")
    collection_start = time.time()


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

    # Collect JMX metrics if endpoints configured
    jmx_data = {}
    if JMX_ENDPOINTS:
        raw_jmx = collect_jmx_metrics()
        # Extract JVM health based on source type
        for name, raw_metrics in raw_jmx.items():
            if JVM_METRICS_SOURCE == 'micrometer':
                jmx_data[name] = extract_jvm_health_metrics_quarkus(raw_metrics)
            else:
                jmx_data[name] = extract_jvm_health_metrics(raw_metrics)

    # Collect node data in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CAMUNDA_NODES)) as executor:
        futures = [
            executor.submit(fetch_node_data, name, url, jmx_data)
            for name, url in CAMUNDA_NODES.items()
        ]
        cluster_metrics = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Sort by node name
    cluster_metrics.sort(key=lambda x: x['name'])

    # Count running nodes and aggregate totals
    for node in cluster_metrics:
        if node["status"] == "RUNNING":
            cluster_status["running_nodes"] += 1
            if not cluster_status["engine_version"]:
                cluster_status["engine_version"] = "7.x"
            # Aggregate node-specific metrics
            totals["jobs_executed_total"] += node["jobs_successful"]
        else:
            cluster_status["issues"].append(
                f"Node {node['url']}: {node.get('error', node['status'])}"
            )

    # Get shared database state from first running node
    first_node = next((n for n in cluster_metrics if n["status"] == "RUNNING"), None)
    if first_node:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                api_calls = {
                    'instances': executor.submit(
                        requests.get, f"{first_node['url']}/process-instance/count",
                        timeout=10, auth=CAMUNDA_AUTH
                    ),
                    'tasks': executor.submit(
                        requests.get, f"{first_node['url']}/task/count",
                        timeout=10, auth=CAMUNDA_AUTH
                    ),
                    'ext_tasks': executor.submit(
                        requests.get, f"{first_node['url']}/external-task/count",
                        timeout=10, auth=CAMUNDA_AUTH
                    ),
                    'incidents': executor.submit(
                        requests.get, f"{first_node['url']}/incident/count",
                        timeout=10, auth=CAMUNDA_AUTH
                    ),
                    'deployments': executor.submit(
                        requests.get, f"{first_node['url']}/deployment/count",
                        timeout=10, auth=CAMUNDA_AUTH
                    ),
                    'process_defs': executor.submit(
                        requests.get, f"{first_node['url']}/process-definition/count",
                        timeout=10, auth=CAMUNDA_AUTH
                    ),
                    'dmn_defs': executor.submit(
                        requests.get, f"{first_node['url']}/decision-definition/count",
                        timeout=10, auth=CAMUNDA_AUTH
                    ),
                    'jobs_total': executor.submit(
                        requests.get, f"{first_node['url']}/job/count",
                        timeout=10, auth=CAMUNDA_AUTH
                    ),
                    'jobs_active': executor.submit(
                        requests.get, f"{first_node['url']}/job/count?executable=true",
                        timeout=10, auth=CAMUNDA_AUTH
                    ),
                    'jobs_failed': executor.submit(
                        requests.get, f"{first_node['url']}/job/count?noRetriesLeft=true",
                        timeout=10, auth=CAMUNDA_AUTH
                    )
                }

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
        except Exception as e:
            logger.error(f"Shared state collection failed: {e}", exc_info=True)

    # Set shared data for all running nodes
    for node in cluster_metrics:
        if node["status"] == "RUNNING":
            node["active_instances"] = totals["active_instances"]
            node["user_tasks"] = totals["user_tasks"]
            node["external_tasks"] = totals["external_tasks"]
            node["incidents"] = totals["incidents"]
            node["deployment_count"] = totals["deployment_count"]
            node["process_definitions"] = totals["process_definitions"]
            node["dmn_definitions"] = totals["dmn_definitions"]

    # Database health check
    db_metrics = {"connectivity": "OK"}
    try:
        db_start = time.time()
        execute_query("SELECT 1")
        db_metrics["latency_ms"] = int((time.time() - db_start) * 1000)

        # Get connection stats
        conn_stats = execute_query(
            "SELECT count(*) AS active, "
            "(SELECT setting::int FROM pg_settings WHERE name='max_connections') AS max "
            "FROM pg_stat_activity WHERE datname=current_database();"
        )
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

    # Calculate jobs per minute (simple average from node totals)
    if totals["jobs_executed_total"] > 0:
        # Estimate based on metric collection interval (assume 1 minute)
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
# Flask Routes
# ============================================================

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        data = collect_engine_health()
        return render_template('index.html', data=data, stuck_days=STUCK_INSTANCE_DAYS)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}", exc_info=True)
        return f"<h1>Error</h1><pre>{e}</pre>", 500


@app.route('/api/health')
def api_health():
    """API endpoint for full health data"""
    try:
        data = collect_engine_health()
        # Convert datetime to ISO format
        data['timestamp'] = data['timestamp'].isoformat()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Lazy Loading API Endpoints
# ============================================================

@app.route('/api/metrics/stuck-instances')
def api_stuck_instances():
    """Get count of stuck instances"""
    try:
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
                AND la.last_update < NOW() - INTERVAL '{STUCK_INSTANCE_DAYS} days'
            ) as stuck
        """)
        return jsonify({
            "value": result[0]['count'] if result else 0,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics/pending-messages')
def api_pending_messages():
    """Get count of pending message subscriptions"""
    try:
        result = execute_query("""
            SELECT COUNT(*) as count 
            FROM act_ru_event_subscr 
            WHERE event_type_ = 'message'
        """)
        return jsonify({
            "value": result[0]['count'] if result else 0,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics/pending-signals')
def api_pending_signals():
    """Get count of pending signal subscriptions"""
    try:
        result = execute_query("""
            SELECT COUNT(*) as count 
            FROM act_ru_event_subscr 
            WHERE event_type_ = 'signal'
        """)
        return jsonify({
            "value": result[0]['count'] if result else 0,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics/job-throughput')
def api_job_throughput():
    """Get job execution throughput (jobs per minute)"""
    try:
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics/database')
def api_database_metrics():
    """Get database storage and performance metrics"""
    try:
        data = collect_database_metrics()
        data["timestamp"] = datetime.now().isoformat()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Prometheus Metrics Export
# ============================================================

@app.route('/metrics')
def prometheus_metrics():
    """Export metrics in Prometheus format"""
    try:
        data = collect_engine_health()

        lines = []
        lines.append("# HELP camunda_cluster_info Cluster information")
        lines.append("# TYPE camunda_cluster_info gauge")

        cluster = data.get('cluster_status', {})
        lines.append(f'camunda_cluster_info{{version="{cluster.get("engine_version", "unknown")}"}} {cluster.get("total_nodes", 0)}')

        lines.append("# HELP camunda_cluster_running_nodes Number of running nodes")
        lines.append("# TYPE camunda_cluster_running_nodes gauge")
        lines.append(f'camunda_cluster_running_nodes {cluster.get("running_nodes", 0)}')

        totals = data.get('totals', {})

        # Core metrics
        lines.append("# HELP camunda_active_instances Active process instances")
        lines.append("# TYPE camunda_active_instances gauge")
        lines.append(f'camunda_active_instances {totals.get("active_instances", 0)}')

        lines.append("# HELP camunda_user_tasks Active user tasks")
        lines.append("# TYPE camunda_user_tasks gauge")
        lines.append(f'camunda_user_tasks {totals.get("user_tasks", 0)}')

        lines.append("# HELP camunda_external_tasks Active external tasks")
        lines.append("# TYPE camunda_external_tasks gauge")
        lines.append(f'camunda_external_tasks {totals.get("external_tasks", 0)}')

        lines.append("# HELP camunda_incidents Active incidents")
        lines.append("# TYPE camunda_incidents gauge")
        lines.append(f'camunda_incidents {totals.get("incidents", 0)}')

        lines.append("# HELP camunda_total_jobs Total jobs")
        lines.append("# TYPE camunda_total_jobs gauge")
        lines.append(f'camunda_total_jobs {totals.get("total_jobs", 0)}')

        lines.append("# HELP camunda_failed_jobs Jobs with no retries left")
        lines.append("# TYPE camunda_failed_jobs gauge")
        lines.append(f'camunda_failed_jobs {totals.get("failed_jobs", 0)}')

        lines.append("# HELP camunda_deployments Total deployments")
        lines.append("# TYPE camunda_deployments gauge")
        lines.append(f'camunda_deployments {totals.get("deployment_count", 0)}')

        lines.append("# HELP camunda_process_definitions Process definitions")
        lines.append("# TYPE camunda_process_definitions gauge")
        lines.append(f'camunda_process_definitions {totals.get("process_definitions", 0)}')

        lines.append("# HELP camunda_dmn_definitions DMN definitions")
        lines.append("# TYPE camunda_dmn_definitions gauge")
        lines.append(f'camunda_dmn_definitions {totals.get("dmn_definitions", 0)}')

        # Per-node metrics
        nodes = data.get('cluster_nodes', [])
        lines.append("# HELP camunda_node_status Node status (1=RUNNING, 0=ERROR/DOWN)")
        lines.append("# TYPE camunda_node_status gauge")
        for node in nodes:
            status = 1 if node.get('status') == 'RUNNING' else 0
            node_name = node.get('name', 'unknown').replace('-', '_')
            lines.append(f'camunda_node_status{{node="{node_name}",url="{node.get("url", "")}"}} {status}')

        lines.append("# HELP camunda_node_response_time_ms Node response time in milliseconds")
        lines.append("# TYPE camunda_node_response_time_ms gauge")
        for node in nodes:
            if node.get('response_time_ms') is not None:
                node_name = node.get('name', 'unknown').replace('-', '_')
                lines.append(f'camunda_node_response_time_ms{{node="{node_name}"}} {node.get("response_time_ms", 0)}')

        lines.append("# HELP camunda_node_job_success_rate Job success rate per node")
        lines.append("# TYPE camunda_node_job_success_rate gauge")
        for node in nodes:
            if node.get('status') == 'RUNNING':
                node_name = node.get('name', 'unknown').replace('-', '_')
                lines.append(f'camunda_node_job_success_rate{{node="{node_name}"}} {node.get("job_success_rate", 100)}')

        lines.append("# HELP camunda_node_workload_score Workload score per node")
        lines.append("# TYPE camunda_node_workload_score gauge")
        for node in nodes:
            if node.get('status') == 'RUNNING':
                node_name = node.get('name', 'unknown').replace('-', '_')
                lines.append(f'camunda_node_workload_score{{node="{node_name}"}} {node.get("workload_score", 0)}')

        # JVM metrics per node
        lines.append("# HELP camunda_jvm_heap_used_mb JVM heap used in MB")
        lines.append("# TYPE camunda_jvm_heap_used_mb gauge")
        for node in nodes:
            jvm = node.get('jvm_metrics', {})
            if jvm.get('status') == 'HEALTHY':
                node_name = node.get('name', 'unknown').replace('-', '_')
                lines.append(f'camunda_jvm_heap_used_mb{{node="{node_name}"}} {jvm.get("memory", {}).get("heap_used_mb", 0)}')

        lines.append("# HELP camunda_jvm_heap_utilization_percent JVM heap utilization percentage")
        lines.append("# TYPE camunda_jvm_heap_utilization_percent gauge")
        for node in nodes:
            jvm = node.get('jvm_metrics', {})
            if jvm.get('status') == 'HEALTHY':
                node_name = node.get('name', 'unknown').replace('-', '_')
                lines.append(f'camunda_jvm_heap_utilization_percent{{node="{node_name}"}} {jvm.get("memory", {}).get("heap_utilization_pct", 0)}')

        lines.append("# HELP camunda_jvm_cpu_load_percent CPU load percentage")
        lines.append("# TYPE camunda_jvm_cpu_load_percent gauge")
        for node in nodes:
            jvm = node.get('jvm_metrics', {})
            if jvm.get('status') == 'HEALTHY':
                node_name = node.get('name', 'unknown').replace('-', '_')
                lines.append(f'camunda_jvm_cpu_load_percent{{node="{node_name}"}} {jvm.get("system", {}).get("cpu_load_pct", 0)}')

        # Database metrics
        db_metrics = data.get('db_metrics', {})
        lines.append("# HELP camunda_db_latency_ms Database query latency in milliseconds")
        lines.append("# TYPE camunda_db_latency_ms gauge")
        lines.append(f'camunda_db_latency_ms {db_metrics.get("latency_ms", 0)}')

        lines.append("# HELP camunda_db_active_connections Active database connections")
        lines.append("# TYPE camunda_db_active_connections gauge")
        lines.append(f'camunda_db_active_connections {db_metrics.get("active_connections", 0)}')

        lines.append("# HELP camunda_db_connection_utilization_percent Connection pool utilization percentage")
        lines.append("# TYPE camunda_db_connection_utilization_percent gauge")
        lines.append(f'camunda_db_connection_utilization_percent {db_metrics.get("connection_utilization", 0)}')

        return '\n'.join(lines) + '\n', 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        logger.error(f"Prometheus metrics export failed: {e}", exc_info=True)
        return f"# ERROR: {str(e)}\n", 500, {'Content-Type': 'text/plain; charset=utf-8'}


# ============================================================
# Custom Jinja Filters
# ============================================================

@app.template_filter('number')
def format_number(value):
    """Format large numbers"""
    if value is None:
        return "0"
    try:
        value = int(value)
        if value < 1000:
            return str(value)
        elif value < 1000000:
            return f"{value / 1000:.1f}K"
        else:
            return f"{value / 1000000:.1f}M"
    except:
        return str(value)


@app.template_filter('duration')
def format_duration(ms):
    """Format duration in milliseconds"""
    if ms is None:
        return "N/A"
    try:
        ms = float(ms)
        if ms < 1000:
            return f"{ms:.0f}ms"
        elif ms < 60000:
            return f"{ms / 1000:.2f}s"
        elif ms < 3600000:
            return f"{ms / 60000:.2f}min"
        else:
            return f"{ms / 3600000:.2f}h"
    except:
        return "N/A"


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting Camunda Health Monitor on port {port}")
    logger.info(f"Dashboard: http://localhost:{port}")
    logger.info(f"API: http://localhost:{port}/api/health")
    logger.info(f"Metrics: http://localhost:{port}/metrics\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
