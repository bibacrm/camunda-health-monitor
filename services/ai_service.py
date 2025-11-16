"""
AI/ML Service
Intelligent analytics and predictions for Camunda cluster health
Uses only historical data from ACT_HI_* and ACT_RU_* tables
"""
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from helpers.error_handler import safe_execute
from helpers.db_helper import execute_query

logger = logging.getLogger('champa_monitor.ai_service')


class AIAnalytics:
    """AI-powered analytics for Camunda cluster"""

    def __init__(self, config=None):
        self.anomaly_detector = None
        self.incident_clusterer = None
        # Don't access current_app during init - it may not be available
        self._config = config

    def _get_config(self, key, default=None):
        """Helper to get config value"""
        # Lazy load config from Flask app if not provided
        if self._config is None:
            try:
                from flask import current_app
                self._config = current_app.config
            except RuntimeError:
                # No app context available, use empty dict
                # This allows the class to be instantiated outside app context
                self._config = {}

        return self._config.get(key, default)

    def get_cluster_health_score(self, cluster_data, db_metrics):
        """
        Calculate composite AI health score (0-100)

        Weighted scoring:
        - 30% JVM Health (heap, CPU, GC)
        - 20% DB Health (latency, connections)
        - 20% Incident Rate
        - 15% Job Success Rate
        - 10% Process Execution Health
        - 5% Cluster Availability
        """
        try:
            scores = {}

            # 1. JVM Health (30%)
            jvm_score = self._calculate_jvm_health(cluster_data.get('cluster_nodes', []))
            scores['jvm'] = jvm_score * 0.30

            # 2. DB Health (20%)
            db_score = self._calculate_db_health(db_metrics)
            scores['db'] = db_score * 0.20

            # 3. Incident Rate (20%)
            incident_score = self._calculate_incident_health(cluster_data.get('totals', {}))
            scores['incidents'] = incident_score * 0.20

            # 4. Job Success Rate (15%)
            job_score = self._calculate_job_health(cluster_data.get('totals', {}))
            scores['jobs'] = job_score * 0.15

            # 5. Process Execution Health (10%)
            process_score = self._calculate_process_health(cluster_data.get('totals', {}))
            scores['process'] = process_score * 0.10

            # 6. Cluster Availability (5%)
            cluster_score = self._calculate_cluster_availability(cluster_data.get('cluster_status', {}))
            scores['cluster'] = cluster_score * 0.05

            total_score = sum(scores.values())

            return {
                'overall_score': round(total_score, 1),
                'breakdown': {k: round(v / self._get_weight(k) * 100, 1) for k, v in scores.items()},
                'grade': self._get_grade(total_score),
                'factors': self._get_health_factors(scores)
            }
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return {
                'overall_score': 0,
                'breakdown': {},
                'grade': 'Unknown',
                'factors': 'Error calculating health score'
            }

    def _get_weight(self, key):
        """Get weight for score component"""
        weights = {
            'jvm': 0.30,
            'db': 0.20,
            'incidents': 0.20,
            'jobs': 0.15,
            'process': 0.10,
            'cluster': 0.05
        }
        return weights.get(key, 1.0)

    def _calculate_jvm_health(self, nodes):
        """Calculate JVM health score from node metrics"""
        if not nodes:
            return 50

        running_nodes = [n for n in nodes if n.get('status') == 'RUNNING']
        if not running_nodes:
            return 0

        scores = []
        for node in running_nodes:
            jvm = node.get('jvm_metrics', {})
            if jvm.get('status') != 'HEALTHY':
                scores.append(0)
                continue

            memory = jvm.get('memory', {})
            system = jvm.get('system', {})
            gc = jvm.get('garbage_collection', {})

            # Heap usage score (lower is better)
            heap_pct = memory.get('heap_utilization_pct', 0)
            heap_score = max(0, 100 - heap_pct * 1.5)  # Penalize high usage

            # CPU score (lower is better)
            cpu_pct = system.get('cpu_load_pct', 0)
            cpu_score = max(0, 100 - cpu_pct * 1.2)

            # GC time score (lower is better)
            gc_time = gc.get('time_ms', 0)
            gc_score = max(0, 100 - (gc_time / 100))

            node_score = (heap_score * 0.5 + cpu_score * 0.3 + gc_score * 0.2)
            scores.append(node_score)

        return np.mean(scores) if scores else 50

    def _calculate_db_health(self, db_metrics):
        """Calculate database health score"""
        if not db_metrics or db_metrics.get('connectivity') != 'OK':
            return 0

        latency = db_metrics.get('latency_ms', 0)
        conn_util = db_metrics.get('connection_utilization', 0)

        # Latency score (penalize high latency)
        if latency < 10:
            latency_score = 100
        elif latency < 50:
            latency_score = 80
        elif latency < 100:
            latency_score = 60
        elif latency < 500:
            latency_score = 40
        else:
            latency_score = 20

        # Connection pool score
        conn_score = max(0, 100 - conn_util * 1.5)

        return latency_score * 0.7 + conn_score * 0.3

    def _calculate_incident_health(self, totals):
        """Calculate health based on incident rate"""
        incidents = totals.get('incidents', 0)
        active_instances = max(totals.get('active_instances', 1), 1)

        incident_rate = (incidents / active_instances) * 100

        if incident_rate == 0:
            return 100
        elif incident_rate < 1:
            return 90
        elif incident_rate < 5:
            return 70
        elif incident_rate < 10:
            return 50
        else:
            return max(0, 50 - incident_rate * 2)

    def _calculate_job_health(self, totals):
        """Calculate job executor health"""
        total_jobs = totals.get('total_jobs', 0)
        failed_jobs = totals.get('failed_jobs', 0)

        if total_jobs == 0:
            return 100

        success_rate = ((total_jobs - failed_jobs) / total_jobs) * 100
        return success_rate

    def _calculate_process_health(self, totals):
        """Calculate process execution health"""
        active = totals.get('active_instances', 0)
        stuck = totals.get('stuck_instances', 0)

        if active == 0:
            return 100

        stuck_rate = (stuck / max(active, 1)) * 100
        return max(0, 100 - stuck_rate * 10)

    def _calculate_cluster_availability(self, cluster_status):
        """Calculate cluster availability score"""
        total = cluster_status.get('total_nodes', 1)
        running = cluster_status.get('running_nodes', 0)

        return (running / total) * 100

    def _get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return 'A - Excellent'
        elif score >= 80:
            return 'B - Good'
        elif score >= 70:
            return 'C - Fair'
        elif score >= 60:
            return 'D - Poor'
        else:
            return 'F - Critical'

    def _get_health_factors(self, scores):
        """Get human-readable health factors"""
        breakdown = {k: round(v / self._get_weight(k) * 100, 1) for k, v in scores.items()}

        factors = []
        if breakdown.get('jvm', 100) < 70:
            factors.append('JVM under pressure')
        if breakdown.get('db', 100) < 70:
            factors.append('Database performance issues')
        if breakdown.get('incidents', 100) < 80:
            factors.append('High incident rate')
        if breakdown.get('jobs', 100) < 90:
            factors.append('Job failures detected')
        if breakdown.get('cluster', 100) < 100:
            factors.append('Node(s) down')

        return ', '.join(factors) if factors else 'All systems healthy'

    def detect_process_anomalies(self, lookback_days=None):
        """
        Detect anomalies in process execution times using historical data
        Returns processes that are behaving abnormally
        """
        try:
            # Use config if not provided
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            max_instances = self._get_config('AI_MAX_INSTANCES', 50000)
            min_data = self._get_config('AI_MIN_DATA', 10)
            zscore_threshold = self._get_config('AI_ZSCORE_THRESHOLD', 1.0)
            critical_duration = self._get_config('AI_CRITICAL_DURATION_SECONDS', 86400)
            high_duration = self._get_config('AI_HIGH_DURATION_SECONDS', 7200)
            medium_duration = self._get_config('AI_MEDIUM_DURATION_SECONDS', 3600)
            slow_avg_duration = self._get_config('AI_SLOW_AVG_DURATION_SECONDS', 600)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            # Get historical process execution data
            query = f"""
                SELECT 
                    proc_def_key_,
                    EXTRACT(EPOCH FROM (end_time_ - start_time_)) * 1000 as duration_ms,
                    end_time_
                FROM act_hi_procinst
                WHERE end_time_ IS NOT NULL
                AND start_time_ IS NOT NULL
                AND end_time_ > start_time_
                AND end_time_ > NOW() - INTERVAL '{lookback_days} days'
                ORDER BY end_time_ DESC
                LIMIT {max_instances}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Detecting process anomalies"
            )

            if not results or len(results) < min_data:
                return {
                    'anomalies': [],
                    'total_analyzed': 0,
                    'message': f'Insufficient historical data for anomaly detection (found {len(results) if results else 0} instances, need at least {min_data})'
                }

            # Group by process definition
            process_groups = defaultdict(list)
            for row in results:
                process_groups[row['proc_def_key_']].append(float(row['duration_ms']))

            anomalies = []
            total_analyzed = 0

            for proc_key, durations in process_groups.items():
                if len(durations) < min_data:
                    continue

                total_analyzed += 1

                # Calculate statistics
                mean_duration = np.mean(durations)
                std_duration = np.std(durations)
                max_duration = np.max(durations)

                # Get recent executions (last 20% for better recent sample)
                recent_count = max(int(len(durations) * 0.2), 3)
                recent_durations = durations[:recent_count]
                recent_mean = np.mean(recent_durations)

                # DUAL DETECTION LOGIC
                detected = False
                severity = 'low'
                anomaly_types = []
                z_score = 0

                # 1. Statistical anomaly detection (Z-score based)
                if std_duration > 0:
                    z_score = abs((recent_mean - mean_duration) / std_duration)

                    if z_score > zscore_threshold:
                        detected = True
                        anomaly_types.append('statistical_deviation')
                        if z_score > 3:
                            severity = 'high'
                        elif z_score > 2:
                            severity = 'medium'
                        else:
                            severity = 'low'

                # 2. Absolute performance anomaly detection (stuck/hanging processes)
                mean_seconds = mean_duration / 1000
                max_seconds = max_duration / 1000

                if max_seconds > medium_duration:
                    detected = True
                    anomaly_types.append('extreme_duration')
                    if max_seconds > critical_duration:
                        severity = 'critical'
                    elif max_seconds > high_duration:
                        severity = 'high'
                    elif severity == 'low':
                        severity = 'medium'

                elif mean_seconds > slow_avg_duration:
                    detected = True
                    anomaly_types.append('slow_average')
                    if severity == 'low':
                        severity = 'medium'

                if detected:
                    deviation_pct = ((recent_mean - mean_duration) / mean_duration * 100) if mean_duration > 0 else 0

                    anomalies.append({
                        'process_key': proc_key,
                        'baseline_avg_ms': round(float(mean_duration), 2),
                        'recent_avg_ms': round(float(recent_mean), 2),
                        'max_duration_ms': round(float(max_duration), 2),
                        'deviation_pct': round(deviation_pct, 1),
                        'z_score': round(float(z_score), 2),
                        'severity': severity,
                        'anomaly_types': anomaly_types,
                        'instances_analyzed': len(durations)
                    })

            # Sort by severity (critical > high > medium > low), then by Z-score
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            anomalies.sort(key=lambda x: (severity_order.get(x['severity'], 99), -x['z_score']))

            # If no anomalies found, provide helpful message
            if len(anomalies) == 0 and total_analyzed > 0:
                # Calculate how many would be detected at lower thresholds
                borderline_count = 0
                for proc_key, durations in process_groups.items():
                    if len(durations) < min_data:
                        continue
                    mean_duration = np.mean(durations)
                    std_duration = np.std(durations)
                    recent_count = max(int(len(durations) * 0.2), 3)
                    recent_mean = np.mean(durations[:recent_count])
                    if std_duration > 0:
                        z_score = abs((recent_mean - mean_duration) / std_duration)
                        if 0.5 < z_score <= zscore_threshold:
                            borderline_count += 1

                message = f'✓ No anomalies detected - All {total_analyzed} processes executing consistently within normal parameters. This indicates stable, predictable performance!'
                if borderline_count > 0:
                    message += f' ({borderline_count} processes show minor variations but within acceptable range)'

                return {
                    'anomalies': [],
                    'total_analyzed': total_analyzed,
                    'detection_window_days': lookback_days,
                    'message': message,
                    'health_status': 'excellent',
                    'borderline_count': borderline_count,
                    'instances_reviewed': len(results)
                }

            return {
                'anomalies': anomalies[:max_results],
                'total_analyzed': total_analyzed,
                'detection_window_days': lookback_days,
                'message': f'Analyzed {total_analyzed} process definitions from {len(results)} instances',
                'instances_reviewed': len(results)
            }

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {
                'anomalies': [],
                'total_analyzed': 0,
                'error': str(e)
            }

    def analyze_incident_patterns(self, lookback_days=None):
        """
        Cluster similar incidents using text analysis
        Identifies common error patterns and root causes
        """
        try:
            # Use config if not provided
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            max_incidents = self._get_config('AI_MAX_INCIDENTS', 1000)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            # First try historical incidents
            query = f"""
                SELECT 
                    incident_type_,
                    incident_msg_,
                    proc_def_key_,
                    activity_id_,
                    create_time_
                FROM act_hi_incident
                WHERE create_time_ > NOW() - INTERVAL '{lookback_days} days'
                ORDER BY create_time_ DESC
                LIMIT {max_incidents}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Analyzing incident patterns"
            )

            # If no historical incidents, try to analyze from runtime jobs with exceptions
            if not results or len(results) < 1:
                logger.info("No historical incidents found, analyzing runtime job exceptions")

                # Fallback: Analyze active jobs with exceptions
                query_fallback = f"""
                    SELECT 
                        type_ as incident_type,
                        exception_msg_ as error_message,
                        process_instance_id_,
                        retries_
                    FROM act_ru_job
                    WHERE exception_msg_ IS NOT NULL
                    AND exception_msg_ != ''
                    LIMIT {max_incidents}
                """

                fallback_results = safe_execute(
                    lambda: execute_query(query_fallback),
                    default_value=[],
                    context="Analyzing job exceptions as incidents"
                )

                if fallback_results and len(fallback_results) > 0:
                    # Group by exception type and message pattern
                    pattern_groups = defaultdict(list)
                    for row in fallback_results:
                        incident_type = row['incident_type'] if row['incident_type'] else 'Unknown Type'
                        # Extract first 100 chars of error message
                        error_msg = row['error_message'][:100] if row['error_message'] else 'No error message'
                        key = f"{incident_type}:::{error_msg}"
                        pattern_groups[key].append(row)

                    patterns = []
                    for key, jobs in pattern_groups.items():
                        parts = key.split(':::', 1)
                        incident_type = parts[0] if len(parts) > 0 else 'Unknown'
                        msg = parts[1] if len(parts) > 1 else 'No message'

                        # Get affected process instances
                        affected_processes = list(set([j['process_instance_id_'] for j in jobs if j.get('process_instance_id_')]))

                        # Calculate retry exhaustion
                        retries_exhausted = len([j for j in jobs if j.get('retries_', 0) == 0])

                        patterns.append({
                            'incident_type': incident_type,
                            'error_message': msg,
                            'occurrence_count': len(jobs),
                            'affected_processes': affected_processes[:5],
                            'affected_activities': [],  # Not available from runtime jobs
                            'first_seen': None,  # Not available from runtime jobs
                            'last_seen': None,  # Not available from runtime jobs
                            'frequency_per_day': None,  # Can't calculate without timestamps
                            'retries_exhausted': retries_exhausted,
                            'source': 'runtime_jobs'
                        })

                    # Sort by occurrence count
                    patterns.sort(key=lambda x: x['occurrence_count'], reverse=True)

                    return {
                        'patterns': patterns[:max_results],
                        'total_incidents': len(fallback_results),
                        'unique_patterns': len(patterns),
                        'analysis_window_days': lookback_days,
                        'message': f'Found {len(patterns)} error patterns from {len(fallback_results)} active jobs with exceptions (historical incidents not logged)',
                        'health_status': 'degraded',
                        'data_source': 'runtime_jobs'
                    }

                # Truly no incidents anywhere
                return {
                    'patterns': [],
                    'total_incidents': 0,
                    'unique_patterns': 0,
                    'analysis_window_days': lookback_days,
                    'message': f'✓ No incidents found in last {lookback_days} days - System is healthy!',
                    'health_status': 'excellent',
                    'data_source': 'historical'
                }

            # Process historical incidents (original logic)
            pattern_groups = defaultdict(list)
            for row in results:
                incident_type = row['incident_type_'] if row['incident_type_'] else 'Unknown Type'
                incident_msg = row['incident_msg_'][:100] if row['incident_msg_'] else 'No error message'
                key = f"{incident_type}:::{incident_msg}"
                pattern_groups[key].append(row)

            patterns = []
            for key, incidents in pattern_groups.items():
                parts = key.split(':::', 1)
                incident_type = parts[0] if len(parts) > 0 else 'Unknown'
                msg = parts[1] if len(parts) > 1 else 'No message'

                affected_processes = list(set([i['proc_def_key_'] for i in incidents if i.get('proc_def_key_')]))
                affected_activities = list(set([i['activity_id_'] for i in incidents if i.get('activity_id_')]))

                timestamps = [i['create_time_'] for i in incidents if i.get('create_time_')]
                first_seen = min(timestamps).isoformat() if timestamps else None
                last_seen = max(timestamps).isoformat() if timestamps else None

                patterns.append({
                    'incident_type': incident_type,
                    'error_message': msg,
                    'occurrence_count': len(incidents),
                    'affected_processes': affected_processes[:5],
                    'affected_activities': affected_activities[:5],
                    'first_seen': first_seen,
                    'last_seen': last_seen,
                    'frequency_per_day': round(len(incidents) / max(lookback_days, 1), 2),
                    'source': 'historical'
                })

            patterns.sort(key=lambda x: x['occurrence_count'], reverse=True)

            return {
                'patterns': patterns[:max_results],
                'total_incidents': len(results),
                'unique_patterns': len(patterns),
                'analysis_window_days': lookback_days,
                'message': f'Found {len(patterns)} unique incident patterns from {len(results)} incidents',
                'data_source': 'historical'
            }

        except Exception as e:
            logger.error(f"Error analyzing incidents: {e}")
            return {
                'patterns': [],
                'total_incidents': 0,
                'error': str(e),
                'data_source': 'error'
            }

    def identify_bottlenecks(self, lookback_days=None):
        """
        Identify process bottlenecks by analyzing activity durations
        """
        try:
            # Use config if not provided
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            max_activities = self._get_config('AI_MAX_INSTANCES', 50000)
            min_data = self._get_config('AI_MIN_DATA', 10)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            query = f"""
                SELECT 
                    proc_def_key_,
                    act_id_ as activity_id,
                    act_name_ as activity_name,
                    EXTRACT(EPOCH FROM (end_time_ - start_time_)) * 1000 as duration_ms,
                    end_time_
                FROM act_hi_actinst
                WHERE end_time_ IS NOT NULL
                AND end_time_ > NOW() - INTERVAL '{lookback_days} days'
                AND act_id_ IS NOT NULL
                ORDER BY end_time_ DESC
                LIMIT {max_activities}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Identifying bottlenecks"
            )

            if not results or len(results) < 20:
                return {
                    'bottlenecks': [],
                    'total_analyzed': 0,
                    'message': 'Insufficient activity data'
                }

            # Group by process + activity
            activity_groups = defaultdict(list)
            for row in results:
                key = f"{row['proc_def_key_']}:{row['activity_id']}"
                activity_groups[key].append({
                    'duration': row['duration_ms'],
                    'name': row['activity_name']
                })

            bottlenecks = []
            for key, activities in activity_groups.items():
                if len(activities) < min_data:
                    continue

                proc_key, activity_id = key.split(':', 1)
                # Convert Decimal to float to avoid type errors
                durations = [float(a['duration']) for a in activities]

                avg_duration = np.mean(durations)
                p95_duration = np.percentile(durations, 95)
                p99_duration = np.percentile(durations, 99)

                # Only include if average duration > 1 second
                if avg_duration > 1000:
                    bottlenecks.append({
                        'process_key': proc_key,
                        'activity_id': activity_id,
                        'activity_name': activities[0]['name'] or activity_id,
                        'avg_duration_ms': round(float(avg_duration), 2),
                        'p95_duration_ms': round(float(p95_duration), 2),
                        'p99_duration_ms': round(float(p99_duration), 2),
                        'executions': len(activities),
                        'impact_hours_per_week': round((float(avg_duration) / 1000 / 3600) * len(activities) * (7 / lookback_days), 2)
                    })

            # Sort by impact
            bottlenecks.sort(key=lambda x: x['impact_hours_per_week'], reverse=True)

            return {
                'bottlenecks': bottlenecks[:max_results],
                'total_activities_analyzed': len(activity_groups),
                'analysis_window_days': lookback_days
            }

        except Exception as e:
            logger.error(f"Error identifying bottlenecks: {e}")
            return {
                'bottlenecks': [],
                'total_analyzed': 0,
                'error': str(e)
            }

    def predict_job_failures(self, lookback_days=None):
        """
        Analyze job failure patterns and predict failure-prone jobs
        """
        try:
            # Use config if not provided
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            max_job_logs = self._get_config('AI_MAX_INSTANCES', 50000)
            min_data = self._get_config('AI_MIN_DATA', 10)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            # First try standard job log approach
            query = f"""
                SELECT 
                    COALESCE(job_def_type_, 'unknown') as job_def_type_,
                    job_def_configuration_,
                    CASE 
                        WHEN job_exception_msg_ IS NOT NULL AND job_exception_msg_ != '' THEN 'failed'
                        ELSE 'success'
                    END as state,
                    timestamp_
                FROM act_hi_job_log
                WHERE timestamp_ > NOW() - INTERVAL '{lookback_days} days'
                ORDER BY timestamp_ DESC
                LIMIT {max_job_logs}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Predicting job failures"
            )

            # If no job logs, try to infer from failed activities
            if not results or len(results) == 0:
                logger.info("No job logs found, analyzing from activity failures")

                # Fallback: analyze failed activities
                query_fallback = f"""
                    SELECT 
                        act_type_ as job_type,
                        COUNT(*) as total,
                        COUNT(CASE WHEN end_time_ IS NULL THEN 1 END) as incomplete
                    FROM act_hi_actinst
                    WHERE start_time_ > NOW() - INTERVAL '{lookback_days} days'
                    AND act_type_ IN ('serviceTask', 'sendTask', 'businessRuleTask', 'scriptTask')
                    GROUP BY act_type_
                    HAVING COUNT(*) >= {min_data}
                """

                fallback_results = safe_execute(
                    lambda: execute_query(query_fallback),
                    default_value=[],
                    context="Analyzing activity-based job patterns"
                )

                if fallback_results:
                    predictions = []
                    for row in fallback_results:
                        total = row['total']
                        incomplete = row['incomplete'] or 0
                        failure_rate = (incomplete / total * 100) if total > 0 else 0

                        predictions.append({
                            'job_type': f"{row['job_type']} (activity-based)",
                            'total_executions': total,
                            'failed_count': incomplete,
                            'success_count': total - incomplete,
                            'failure_rate_pct': round(failure_rate, 2),
                            'risk_level': 'high' if failure_rate > 10 else 'medium' if failure_rate > 5 else 'low',
                            'recommendation': self._get_job_recommendation(failure_rate)
                        })

                    predictions.sort(key=lambda x: x['failure_rate_pct'], reverse=True)

                    return {
                        'predictions': predictions[:max_results],
                        'total_jobs_analyzed': len(fallback_results),
                        'analysis_window_days': lookback_days,
                        'total_executions': sum(r['total'] for r in fallback_results),
                        'message': f'Analyzed {len(fallback_results)} activity types from process execution (job logs unavailable)',
                        'data_source': 'activity_based'
                    }

                # Truly no data
                return {
                    'predictions': [],
                    'total_analyzed': 0,
                    'message': f'No job log data available. Your Camunda configuration may not log job history, or processes may not use async jobs. This is normal for sync-only processes.',
                    'data_source': 'none'
                }

            # Process job log results (original logic)
            job_groups = defaultdict(lambda: {'total': 0, 'failed': 0, 'success': 0})
            for row in results:
                job_type = row['job_def_type_'] if row['job_def_type_'] else 'unknown'
                job_groups[job_type]['total'] += 1

                if row['state'] == 'failed':
                    job_groups[job_type]['failed'] += 1
                else:
                    job_groups[job_type]['success'] += 1

            predictions = []
            for job_type, stats in job_groups.items():
                if stats['total'] < min_data:
                    continue

                failure_rate = (stats['failed'] / stats['total']) * 100

                predictions.append({
                    'job_type': job_type,
                    'total_executions': stats['total'],
                    'failed_count': stats['failed'],
                    'success_count': stats['success'],
                    'failure_rate_pct': round(failure_rate, 2),
                    'risk_level': 'high' if failure_rate > 10 else 'medium' if failure_rate > 5 else 'low',
                    'recommendation': self._get_job_recommendation(failure_rate)
                })

            predictions.sort(key=lambda x: x['failure_rate_pct'], reverse=True)

            return {
                'predictions': predictions[:max_results],
                'total_jobs_analyzed': len(job_groups),
                'analysis_window_days': lookback_days,
                'total_executions': len(results),
                'message': f'Analyzed {len(job_groups)} job types from {len(results)} executions',
                'data_source': 'job_log'
            }

        except Exception as e:
            logger.error(f"Error predicting job failures: {e}")
            return {
                'predictions': [],
                'total_analyzed': 0,
                'error': str(e),
                'data_source': 'error'
            }

    def _get_job_recommendation(self, failure_rate):
        """Get recommendation based on job failure rate"""
        if failure_rate > 20:
            return 'Critical: Review job configuration and increase retry attempts'
        elif failure_rate > 10:
            return 'Warning: Consider adding error handling or increasing timeouts'
        elif failure_rate > 5:
            return 'Monitor: Review occasional failures for patterns'
        else:
            return 'Healthy: Job executing within normal parameters'

    def analyze_node_performance(self, cluster_nodes):
        """
        Rank nodes by performance and identify underperformers
        """
        try:
            if not cluster_nodes:
                return {
                    'rankings': [],
                    'message': 'No node data available'
                }

            running_nodes = [n for n in cluster_nodes if n.get('status') == 'RUNNING']

            if len(running_nodes) < 2:
                return {
                    'rankings': [],
                    'message': 'Need at least 2 running nodes for comparison'
                }

            rankings = []
            for node in running_nodes:
                jvm = node.get('jvm_metrics', {})

                if jvm.get('status') != 'HEALTHY':
                    continue

                memory = jvm.get('memory', {})
                system = jvm.get('system', {})
                gc = jvm.get('garbage_collection', {})

                # Calculate performance score
                heap_score = max(0, 100 - memory.get('heap_utilization_pct', 0) * 1.2)
                cpu_score = max(0, 100 - system.get('cpu_load_pct', 0) * 1.5)
                gc_score = max(0, 100 - (gc.get('time_ms', 0) / 10))
                thread_score = 100 - (jvm.get('threads', {}).get('current', 0) /
                                     max(jvm.get('threads', {}).get('peak', 1), 1) * 100)

                performance_score = (
                    heap_score * 0.35 +
                    cpu_score * 0.35 +
                    gc_score * 0.20 +
                    thread_score * 0.10
                )

                rankings.append({
                    'node_name': node['name'],
                    'performance_score': round(performance_score, 1),
                    'heap_utilization_pct': memory.get('heap_utilization_pct', 0),
                    'cpu_load_pct': system.get('cpu_load_pct', 0),
                    'gc_time_ms': gc.get('time_ms', 0),
                    'thread_count': jvm.get('threads', {}).get('current', 0),
                    'recommendation': self._get_node_recommendation(performance_score)
                })

            # Sort by performance score
            rankings.sort(key=lambda x: x['performance_score'], reverse=True)

            # Add rank
            for i, ranking in enumerate(rankings):
                ranking['rank'] = i + 1

            return {
                'rankings': rankings,
                'best_performer': rankings[0]['node_name'] if rankings else None,
                'worst_performer': rankings[-1]['node_name'] if rankings else None
            }

        except Exception as e:
            logger.error(f"Error analyzing node performance: {e}")
            return {
                'rankings': [],
                'error': str(e)
            }

    def _get_node_recommendation(self, score):
        """Get recommendation based on node performance score"""
        if score >= 80:
            return 'Optimal performance'
        elif score >= 60:
            return 'Monitor resource usage'
        elif score >= 40:
            return 'Consider restarting or scaling'
        else:
            return 'Critical: Immediate action required'

    def get_process_leaderboard(self, lookback_days=None):
        """
        Performance leaderboard for process definitions
        """
        try:
            # Use config if not provided
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            min_data = self._get_config('AI_MIN_DATA', 10)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            query = f"""
                SELECT 
                    proc_def_key_,
                    COUNT(*) as instance_count,
                    AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_)) * 1000) as avg_duration_ms,
                    COUNT(*) FILTER (WHERE end_time_ IS NOT NULL) as completed_count,
                    COUNT(*) FILTER (WHERE delete_reason_ LIKE '%incident%') as failed_count
                FROM act_hi_procinst
                WHERE start_time_ > NOW() - INTERVAL '{lookback_days} days'
                GROUP BY proc_def_key_
                HAVING COUNT(*) >= {min_data}
                ORDER BY instance_count DESC
                LIMIT {max_results}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Getting process leaderboard"
            )

            if not results:
                return {
                    'leaderboard': [],
                    'message': 'No process data available'
                }

            leaderboard = []
            for row in results:
                completion_rate = (row['completed_count'] / row['instance_count'] * 100) if row['instance_count'] > 0 else 0
                failure_rate = (row['failed_count'] / row['instance_count'] * 100) if row['instance_count'] > 0 else 0

                leaderboard.append({
                    'process_key': row['proc_def_key_'],
                    'instance_count': row['instance_count'],
                    'avg_duration_ms': round(float(row['avg_duration_ms']) if row['avg_duration_ms'] else 0, 2),
                    'completion_rate_pct': round(float(completion_rate), 1),
                    'failure_rate_pct': round(float(failure_rate), 1),
                    'grade': self._get_process_grade(completion_rate, failure_rate)
                })

            return {
                'leaderboard': leaderboard,
                'total_processes': len(leaderboard),
                'analysis_window_days': lookback_days
            }

        except Exception as e:
            logger.error(f"Error getting process leaderboard: {e}")
            return {
                'leaderboard': [],
                'error': str(e)
            }

    def _get_process_grade(self, completion_rate, failure_rate):
        """Grade process performance"""
        if completion_rate >= 95 and failure_rate < 1:
            return 'A'
        elif completion_rate >= 90 and failure_rate < 5:
            return 'B'
        elif completion_rate >= 80 and failure_rate < 10:
            return 'C'
        elif completion_rate >= 70:
            return 'D'
        else:
            return 'F'

    def predict_sla_breaches(self, threshold_hours=None):
        """
        Predict which active tasks are likely to breach SLA
        """
        try:
            # Use config if not provided
            if threshold_hours is None:
                threshold_hours = self._get_config('SLA_THRESHOLD_HOURS', 24)

            warning_threshold_pct = self._get_config('SLA_WARNING_THRESHOLD_PCT', 70)
            max_tasks = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            # Calculate warning threshold in hours
            warning_hours = threshold_hours * (warning_threshold_pct / 100)

            query = f"""
                SELECT 
                    t.id_,
                    t.name_,
                    t.proc_def_id_,
                    t.create_time_,
                    EXTRACT(EPOCH FROM (NOW() - t.create_time_)) / 3600 as age_hours
                FROM act_ru_task t
                WHERE t.create_time_ < NOW() - INTERVAL '{warning_hours} hours'
                ORDER BY t.create_time_
                LIMIT {max_tasks}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Predicting SLA breaches"
            )

            if not results:
                return {
                    'at_risk_tasks': [],
                    'total_analyzed': 0
                }

            # Get historical completion times for similar tasks
            at_risk = []
            for task in results:
                # Simple heuristic: if task is > warning_threshold_pct of threshold, flag it
                # Convert Decimal to float to avoid type errors
                age_hours = float(task['age_hours'])
                risk_pct = (age_hours / threshold_hours) * 100

                if risk_pct > warning_threshold_pct:
                    at_risk.append({
                        'task_id': task['id_'],
                        'task_name': task['name_'],
                        'process_def_id': task['proc_def_id_'],
                        'age_hours': round(age_hours, 2),
                        'risk_pct': round(risk_pct, 1),
                        'time_until_breach_hours': round(threshold_hours - age_hours, 2),
                        'severity': 'critical' if risk_pct > 90 else 'high' if risk_pct > 80 else 'medium'
                    })

            at_risk.sort(key=lambda x: x['risk_pct'], reverse=True)

            return {
                'at_risk_tasks': at_risk[:self._get_config('AI_UI_RESULTS_LIMIT', 20)],
                'total_analyzed': len(results),
                'sla_threshold_hours': threshold_hours,
                'warning_threshold_pct': warning_threshold_pct
            }

        except Exception as e:
            logger.error(f"Error predicting SLA breaches: {e}")
            return {
                'at_risk_tasks': [],
                'error': str(e)
            }

    def find_stuck_activities_smart(self, lookback_days=None):
        """
        Advanced stuck activity detection using statistical thresholds
        Identifies activities taking abnormally long based on historical percentiles
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            percentile = self._get_config('AI_STUCK_ACTIVITY_PERCENTILE', 95)
            multiplier = self._get_config('AI_STUCK_ACTIVITY_MULTIPLIER', 2.0)
            max_activities = self._get_config('AI_MAX_INSTANCES', 50000)
            min_data = self._get_config('AI_MIN_DATA', 10)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            # Build activity duration statistics
            query = f"""
                WITH activity_durations AS (
                    SELECT 
                        proc_def_key_,
                        act_id_,
                        act_name_,
                        EXTRACT(EPOCH FROM (COALESCE(end_time_, NOW()) - start_time_)) as duration_seconds
                    FROM act_hi_actinst
                    WHERE start_time_ > NOW() - INTERVAL '{lookback_days * 3} days'
                    AND act_id_ IS NOT NULL
                ),
                activity_stats AS (
                    SELECT 
                        proc_def_key_,
                        act_id_,
                        act_name_,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_seconds) as median_duration,
                        PERCENTILE_CONT({percentile / 100.0}) WITHIN GROUP (ORDER BY duration_seconds) as p95_duration,
                        AVG(duration_seconds) as avg_duration,
                        STDDEV(duration_seconds) as stddev_duration,
                        COUNT(*) as execution_count
                    FROM activity_durations
                    GROUP BY proc_def_key_, act_id_, act_name_
                    HAVING COUNT(*) >= {min_data}
                )
                SELECT 
                    ai.id_ as activity_instance_id,
                    ai.proc_inst_id_,
                    ai.proc_def_key_,
                    ai.act_id_,
                    ai.act_name_,
                    EXTRACT(EPOCH FROM (NOW() - ai.start_time_)) as current_duration,
                    s.median_duration,
                    s.p95_duration,
                    s.avg_duration,
                    s.stddev_duration,
                    s.execution_count,
                    EXTRACT(EPOCH FROM (NOW() - ai.start_time_)) / NULLIF(s.p95_duration, 0) as duration_ratio
                FROM act_hi_actinst ai
                JOIN activity_stats s ON ai.proc_def_key_ = s.proc_def_key_ AND ai.act_id_ = s.act_id_
                WHERE ai.end_time_ IS NULL
                AND EXTRACT(EPOCH FROM (NOW() - ai.start_time_)) > s.p95_duration * {multiplier}
                ORDER BY duration_ratio DESC
                LIMIT {max_results}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Finding stuck activities (smart)"
            )

            if not results:
                return {
                    'stuck_activities': [],
                    'total_found': 0,
                    'message': f'✓ No stuck activities detected - All activities completing within normal timeframes (P{percentile})',
                    'analysis_window_days': lookback_days
                }

            stuck_activities = []
            for row in results:
                current_duration = float(row['current_duration'])
                p95_duration = float(row['p95_duration']) if row['p95_duration'] else 0
                duration_ratio = float(row['duration_ratio']) if row['duration_ratio'] else 0

                # Calculate Z-score if we have stddev
                z_score = None
                if row['stddev_duration'] and float(row['stddev_duration']) > 0:
                    avg = float(row['avg_duration'])
                    stddev = float(row['stddev_duration'])
                    z_score = (current_duration - avg) / stddev

                severity = 'critical' if duration_ratio > 5 else 'high' if duration_ratio > 3 else 'medium'

                stuck_activities.append({
                    'activity_instance_id': row['activity_instance_id'],
                    'process_instance_id': row['proc_inst_id_'],
                    'process_key': row['proc_def_key_'],
                    'activity_id': row['act_id_'],
                    'activity_name': row['act_name_'] or row['act_id_'],
                    'stuck_for_seconds': round(current_duration, 2),
                    'stuck_for_hours': round(current_duration / 3600, 2),
                    'expected_p95_seconds': round(p95_duration, 2),
                    'median_duration_seconds': round(float(row['median_duration']), 2) if row['median_duration'] else 0,
                    'duration_ratio': round(duration_ratio, 1),
                    'z_score': round(z_score, 2) if z_score else None,
                    'severity': severity,
                    'historical_executions': row['execution_count'],
                    'message': f"Activity taking {duration_ratio:.1f}x longer than P{percentile} of similar executions"
                })

            return {
                'stuck_activities': stuck_activities,
                'total_found': len(stuck_activities),
                'threshold_percentile': percentile,
                'threshold_multiplier': multiplier,
                'analysis_window_days': lookback_days,
                'message': f'Found {len(stuck_activities)} activities significantly exceeding normal duration'
            }

        except Exception as e:
            logger.error(f"Error finding stuck activities: {e}")
            return {
                'stuck_activities': [],
                'total_found': 0,
                'error': str(e)
            }

    def predict_process_duration(self, process_def_key: str, instance_variables: Optional[Dict] = None):
        """
        Predict duration for a running or new process instance
        Uses statistical analysis of historical patterns - more reliable than ML for process execution times
        """
        try:
            training_days = self._get_config('AI_CAPACITY_TRAINING_DAYS', 90)
            min_data = self._get_config('AI_MIN_DATA', 10)

            # Get historical completed instances
            query = f"""
                SELECT 
                    EXTRACT(EPOCH FROM (pi.end_time_ - pi.start_time_)) as duration_seconds,
                    pi.start_time_,
                    pi.end_time_
                FROM act_hi_procinst pi
                WHERE pi.end_time_ IS NOT NULL
                AND pi.start_time_ > NOW() - INTERVAL '{training_days} days'
                AND pi.proc_def_key_ = '{process_def_key}'
                AND pi.end_time_ > pi.start_time_
                ORDER BY pi.end_time_ DESC
                LIMIT 5000
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Predicting process duration"
            )

            instance_count = len(results) if results else 0

            # Convert all Decimal types to float
            if results:
                for r in results:
                    if 'duration_seconds' in r and r['duration_seconds'] is not None:
                        r['duration_seconds'] = float(r['duration_seconds'])

            if not results or len(results) < min_data:
                return {
                    'predicted_duration_ms': None,
                    'predicted_duration_seconds': None,
                    'predicted_duration_hours': None,
                    'confidence': 0,
                    'confidence_pct': 0,
                    'message': f'Insufficient data (found {instance_count}, need {min_data})',
                    'model_type': 'none',
                    'instance_count': instance_count,
                    'percentiles': None
                }

            # Extract durations
            durations = np.array([r['duration_seconds'] for r in results if r.get('duration_seconds') is not None])

            if len(durations) == 0:
                return {
                    'predicted_duration_ms': None,
                    'predicted_duration_seconds': None,
                    'predicted_duration_hours': None,
                    'confidence': 0,
                    'confidence_pct': 0,
                    'message': 'No valid duration data',
                    'model_type': 'none',
                    'instance_count': instance_count,
                    'percentiles': None
                }

            # Calculate statistical measures
            mean_duration = np.mean(durations)
            median_duration = np.median(durations)
            std_duration = np.std(durations)
            p50 = np.percentile(durations, 50)
            p75 = np.percentile(durations, 75)
            p95 = np.percentile(durations, 95)

            # Calculate coefficient of variation (relative variability)
            cv = (std_duration / mean_duration) if mean_duration > 0 else float('inf')

            # Analyze recent trend (last 20% vs overall)
            recent_count = max(int(len(durations) * 0.2), 3)
            recent_durations = durations[:recent_count]
            recent_median = np.median(recent_durations)

            # Determine best prediction method based on data characteristics
            # For Camunda processes:
            # - Low variance (CV < 0.3): Use median (most stable)
            # - Medium variance (0.3 <= CV < 1.0): Use weighted average of median and P75
            # - High variance (CV >= 1.0): Use P75 (safer estimate)

            if cv < 0.3:
                # Stable process - median is best predictor
                predicted_duration = median_duration
                model_type = 'statistical_median'
                confidence_pct = 85
                message = f'Stable process (CV: {cv:.2f}), using median of {instance_count} instances'
            elif cv < 1.0:
                # Moderate variance - use weighted average
                predicted_duration = median_duration * 0.6 + p75 * 0.4
                model_type = 'statistical_weighted'
                confidence_pct = 70
                message = f'Moderate variance (CV: {cv:.2f}), using weighted median-P75 of {instance_count} instances'
            else:
                # High variance - use P75 for safety
                predicted_duration = p75
                model_type = 'statistical_p75'
                confidence_pct = 60
                message = f'High variance (CV: {cv:.2f}), using P75 of {instance_count} instances'

            # Adjust prediction based on recent trend
            trend_factor = (recent_median - median_duration) / median_duration if median_duration > 0 else 0
            if abs(trend_factor) > 0.2:  # Significant trend detected
                predicted_duration = predicted_duration * (1 + trend_factor * 0.5)  # Apply 50% of trend
                message += f' (recent trend: {trend_factor*100:+.1f}%)'

            # Calculate confidence based on sample size and variance
            sample_confidence = min(100.0, float(instance_count))  # Max at 100 samples
            variance_penalty = max(0.0, 30.0 - (cv * 20))  # Penalize high variance
            final_confidence = min(float(confidence_pct), (sample_confidence + variance_penalty) / 2)

            return {
                'predicted_duration_ms': round(predicted_duration * 1000, 2),
                'predicted_duration_seconds': round(predicted_duration, 2),
                'predicted_duration_hours': round(predicted_duration / 3600, 2),
                'confidence': round(final_confidence / 100, 3),
                'confidence_pct': round(final_confidence, 1),
                'message': message,
                'model_type': model_type,
                'instance_count': instance_count,
                'percentiles': {
                    'p50_ms': round(p50 * 1000, 2),
                    'p75_ms': round(p75 * 1000, 2),
                    'p95_ms': round(p95 * 1000, 2),
                    'p50': round(p50 / 3600, 2),
                    'p75': round(p75 / 3600, 2),
                    'p95': round(p95 / 3600, 2)
                },
                'statistics': {
                    'mean_ms': round(mean_duration * 1000, 2),
                    'median_ms': round(median_duration * 1000, 2),
                    'std_ms': round(std_duration * 1000, 2),
                    'mean_hours': round(mean_duration / 3600, 2),
                    'median_hours': round(median_duration / 3600, 2),
                    'std_hours': round(std_duration / 3600, 2),
                    'coefficient_of_variation': round(cv, 2),
                    'recent_trend_pct': round(trend_factor * 100, 1)
                }
            }

        except Exception as e:
            logger.error(f"Error predicting process duration: {e}")
            return {
                'predicted_duration_ms': None,
                'predicted_duration_seconds': None,
                'predicted_duration_hours': None,
                'confidence': 0,
                'confidence_pct': 0,
                'instance_count': 0,
                'percentiles': None,
                'error': str(e),
                'model_type': 'error'
            }

    def forecast_capacity(self, lookback_days=None, forecast_days=None):
        """
        Forecast future capacity needs based on historical load patterns
        Uses time series analysis and trend detection
        """
        try:
            from scipy import stats

            if lookback_days is None:
                lookback_days = self._get_config('AI_CAPACITY_TRAINING_DAYS', 90)
            if forecast_days is None:
                forecast_days = self._get_config('AI_CAPACITY_FORECAST_DAYS', 30)

            # Get historical instance creation patterns
            query = f"""
                SELECT 
                    DATE_TRUNC('hour', start_time_) as hour,
                    proc_def_key_,
                    COUNT(*) as instance_count,
                    AVG(EXTRACT(EPOCH FROM (COALESCE(end_time_, NOW()) - start_time_))) as avg_duration
                FROM act_hi_procinst
                WHERE start_time_ > NOW() - INTERVAL '{lookback_days} days'
                GROUP BY DATE_TRUNC('hour', start_time_), proc_def_key_
                ORDER BY hour DESC
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Forecasting capacity"
            )

            if not results or len(results) < 24:
                return {
                    'forecast': [],
                    'patterns': {},
                    'message': 'Insufficient historical data for capacity forecasting',
                    'growth_rate': None
                }

            df = pd.DataFrame(results)
            df['hour_dt'] = pd.to_datetime(df['hour'])
            df['hour_of_day'] = df['hour_dt'].dt.hour
            df['day_of_week'] = df['hour_dt'].dt.dayofweek
            df['days_since_start'] = (df['hour_dt'] - df['hour_dt'].min()).dt.total_seconds() / 86400

            # Calculate daily totals for better trend analysis
            daily_totals = df.groupby(df['hour_dt'].dt.date)['instance_count'].sum().reset_index()
            daily_totals.columns = ['date', 'daily_count']
            daily_totals['days_since_start'] = range(len(daily_totals))

            # Analyze patterns (hourly for peak detection)
            hourly_pattern = df.groupby('hour_of_day')['instance_count'].mean().to_dict()
            weekly_pattern = df.groupby('day_of_week')['instance_count'].mean().to_dict()

            # Calculate trend using DAILY data for more stable forecast
            if len(daily_totals) > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    daily_totals['days_since_start'].values,
                    daily_totals['daily_count'].values
                )
            else:
                slope, intercept, r_value = 0, daily_totals['daily_count'].mean(), 0

            # Get current average daily load
            avg_daily_load = daily_totals['daily_count'].mean()

            # Forecast next N days
            forecast = []
            current_day = len(daily_totals)

            for day in range(1, forecast_days + 1):
                # Predict based on daily trend
                predicted_daily_load = max(0, intercept + slope * (current_day + day))

                forecast.append({
                    'day': day,
                    'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'predicted_instances': int(predicted_daily_load),
                    'trend': 'increasing' if slope > 1 else 'decreasing' if slope < -1 else 'stable'
                })

            # Identify peak times
            peak_hours = sorted(hourly_pattern.items(), key=lambda x: x[1], reverse=True)[:3]
            busy_days = sorted(weekly_pattern.items(), key=lambda x: x[1], reverse=True)[:3]

            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            return {
                'forecast': forecast,
                'growth_rate_per_day': round(float(slope), 2),
                'trend_confidence': round(float(r_value ** 2), 3),
                'current_avg_daily_load': round(float(avg_daily_load), 0),
                'patterns': {
                    'peak_hours': [{'hour': h, 'avg_instances': round(v, 1)} for h, v in peak_hours],
                    'busiest_days': [{'day': day_names[d], 'avg_instances': round(v, 1)} for d, v in busy_days]
                },
                'analysis_window_days': lookback_days,
                'message': f'Forecast based on {len(daily_totals)} days of data (analyzed {len(df)} hourly data points)'
            }

        except Exception as e:
            logger.error(f"Error forecasting capacity: {e}")
            return {
                'forecast': [],
                'patterns': {},
                'error': str(e)
            }

    def analyze_variable_impact(self, process_def_key: str, variable_names: Optional[List[str]] = None):
        """
        Analyze which process variables correlate with failures or performance
        Identifies variables that impact process outcomes
        """
        try:
            lookback_days = self._get_config('AI_CAPACITY_TRAINING_DAYS', 90)
            min_data = self._get_config('AI_MIN_DATA', 10)
            min_impact = self._get_config('AI_MIN_VARIABLE_IMPACT_PCT', 10)

            # Get instance outcomes
            query = f"""
                WITH instance_outcomes AS (
                    SELECT 
                        pi.proc_inst_id_,
                        pi.proc_def_key_,
                        EXTRACT(EPOCH FROM (pi.end_time_ - pi.start_time_)) as duration,
                        CASE 
                            WHEN pi.delete_reason_ LIKE '%incident%' OR pi.delete_reason_ LIKE '%fail%' THEN 'failed'
                            WHEN pi.end_time_ IS NOT NULL THEN 'completed'
                            ELSE 'running'
                        END as outcome
                    FROM act_hi_procinst pi
                    WHERE pi.start_time_ > NOW() - INTERVAL '{lookback_days} days'
                    AND pi.proc_def_key_ = '{process_def_key}'
                )
                SELECT 
                    v.name_ as variable_name,
                    v.type_ as variable_type,
                    COALESCE(v.text_, v.long_::text, v.double_::text, 'null') as variable_value,
                    io.outcome,
                    COUNT(*) as instance_count,
                    AVG(io.duration) as avg_duration,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY io.duration) as median_duration
                FROM act_hi_varinst v
                JOIN instance_outcomes io ON v.proc_inst_id_ = io.proc_inst_id_
                GROUP BY v.name_, v.type_, variable_value, io.outcome
                HAVING COUNT(*) >= {min_data}
                ORDER BY v.name_, io.outcome
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Analyzing variable impact"
            )

            if not results or len(results) < 2:
                return {
                    'variable_impacts': [],
                    'total_analyzed': 0,
                    'message': f'Insufficient variable data for process {process_def_key}',
                    'process_def_key': process_def_key
                }

            # Analyze each variable
            df = pd.DataFrame(results)
            variable_impacts = []

            for var_name in df['variable_name'].unique():
                var_data = df[df['variable_name'] == var_name]

                failed = var_data[var_data['outcome'] == 'failed']
                completed = var_data[var_data['outcome'] == 'completed']

                if len(failed) == 0 or len(completed) == 0:
                    continue

                total_failed = failed['instance_count'].sum()
                total_completed = completed['instance_count'].sum()
                total_instances = total_failed + total_completed

                failure_rate = (total_failed / total_instances) * 100

                # Duration impact
                failed_avg_duration = failed['avg_duration'].mean()
                completed_avg_duration = completed['avg_duration'].mean()

                if completed_avg_duration > 0:
                    duration_impact = ((failed_avg_duration - completed_avg_duration) / completed_avg_duration) * 100
                else:
                    duration_impact = 0

                # Only report significant impacts
                if abs(failure_rate) > min_impact or abs(duration_impact) > min_impact:
                    impact_level = 'high' if (failure_rate > 20 or abs(duration_impact) > 50) else 'medium'

                    variable_impacts.append({
                        'variable_name': var_name,
                        'variable_type': var_data['variable_type'].iloc[0],
                        'total_instances': int(total_instances),
                        'failure_rate_pct': round(failure_rate, 1),
                        'duration_impact_pct': round(duration_impact, 1),
                        'impact_level': impact_level,
                        'recommendation': self._get_variable_recommendation(var_name, failure_rate, duration_impact),
                        'sample_values': list(var_data['variable_value'].unique()[:5])
                    })

            # Sort by impact
            variable_impacts.sort(key=lambda x: abs(x['failure_rate_pct']) + abs(x['duration_impact_pct']), reverse=True)

            return {
                'variable_impacts': variable_impacts[:self._get_config('AI_UI_RESULTS_LIMIT', 20)],
                'total_analyzed': len(df['variable_name'].unique()),
                'process_def_key': process_def_key,
                'analysis_window_days': lookback_days,
                'message': f'Analyzed {len(df["variable_name"].unique())} variables, found {len(variable_impacts)} with significant impact'
            }

        except Exception as e:
            logger.error(f"Error analyzing variable impact: {e}")
            return {
                'variable_impacts': [],
                'total_analyzed': 0,
                'error': str(e)
            }

    def _get_variable_recommendation(self, var_name: str, failure_rate: float, duration_impact: float) -> str:
        """Generate recommendation based on variable impact"""
        if failure_rate > 20:
            return f'Critical: Variable {var_name} strongly correlates with failures - investigate business logic'
        elif failure_rate > 10:
            return f'Warning: Variable {var_name} correlates with failures - review validation rules'
        elif abs(duration_impact) > 50:
            return f'Performance: Variable {var_name} significantly impacts duration - consider optimization'
        elif abs(duration_impact) > 20:
            return f'Monitor: Variable {var_name} affects duration - track for patterns'
        else:
            return f'Informational: Variable {var_name} shows measurable impact'

    def get_ai_recommendations(self, analysis_results):
        """
        Generate actionable AI recommendations based on all analysis
        """
        recommendations = []

        try:
            # Check anomalies
            if analysis_results.get('anomalies', {}).get('anomalies'):
                count = len(analysis_results['anomalies']['anomalies'])
                recommendations.append({
                    'priority': 'high',
                    'category': 'performance',
                    'message': f'{count} process(es) showing abnormal execution times - investigate recent deployments',
                    'action': 'Review process definitions with anomalies'
                })

            # Check bottlenecks
            bottlenecks = analysis_results.get('bottlenecks', {}).get('bottlenecks', [])
            if bottlenecks and bottlenecks[0].get('impact_hours_per_week', 0) > 10:
                top = bottlenecks[0]
                recommendations.append({
                    'priority': 'high',
                    'category': 'optimization',
                    'message': f'Activity "{top["activity_name"]}" consuming {top["impact_hours_per_week"]}h/week - optimize or parallelize',
                    'action': 'Optimize bottleneck activities'
                })

            # Check incident patterns
            patterns = analysis_results.get('incidents', {}).get('patterns', [])
            if patterns and patterns[0].get('occurrence_count', 0) > 10:
                top = patterns[0]
                recommendations.append({
                    'priority': 'high',
                    'category': 'reliability',
                    'message': f'Recurring incident: "{top["incident_type"]}" ({top["occurrence_count"]} times) - implement permanent fix',
                    'action': 'Fix recurring incidents'
                })

            # Check job failures
            job_predictions = analysis_results.get('job_failures', {}).get('predictions', [])
            high_risk_jobs = [j for j in job_predictions if j.get('risk_level') == 'high']
            if high_risk_jobs:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'jobs',
                    'message': f'{len(high_risk_jobs)} job type(s) with high failure rates - review retry configuration',
                    'action': 'Configure job retries'
                })

            # Check node performance
            node_rankings = analysis_results.get('node_performance', {}).get('rankings', [])
            underperformers = [n for n in node_rankings if n.get('performance_score', 100) < 50]
            if underperformers:
                recommendations.append({
                    'priority': 'high',
                    'category': 'infrastructure',
                    'message': f'{len(underperformers)} node(s) underperforming - consider restart or resource allocation',
                    'action': 'Restart underperforming nodes'
                })

            # Check SLA risks
            sla_risks = analysis_results.get('sla_predictions', {}).get('at_risk_tasks', [])
            critical_risks = [t for t in sla_risks if t.get('severity') == 'critical']
            if critical_risks:
                recommendations.append({
                    'priority': 'critical',
                    'category': 'sla',
                    'message': f'{len(critical_risks)} task(s) about to breach SLA - immediate escalation needed',
                    'action': 'Escalate at-risk tasks'
                })

            # Default recommendation
            if not recommendations:
                recommendations.append({
                    'priority': 'low',
                    'category': 'general',
                    'message': 'System operating within normal parameters - continue monitoring',
                    'action': 'No action required'
                })

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")

        max_recommendations = self._get_config('AI_UI_RESULTS_LIMIT', 20)
        return recommendations[:max_recommendations]


# Singleton instance
_ai_analytics = AIAnalytics()


def get_ai_analytics():
    """Get the AI analytics singleton"""
    return _ai_analytics

