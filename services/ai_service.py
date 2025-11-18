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
        self._config = config

    def _get_config(self, key, default=None):
        """Helper to get config value"""
        if self._config is None:
            try:
                from flask import current_app
                self._config = current_app.config
            except RuntimeError:
                self._config = {}
        return self._config.get(key, default)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _classify_process_category(self, median_hours: float) -> str:
        """Classify process into performance category based on median duration"""
        thresholds = self._get_config('PROCESS_CATEGORY_THRESHOLDS', {
            'ultra_fast': 5/3600,
            'very_fast': 0.5/60,
            'fast_background': 0.1,
            'standard': 0.5,
            'extended': 4,
            'long_running': 24,
            'batch_manual': float('inf')
        })

        for category in ['ultra_fast', 'very_fast', 'fast_background', 'standard',
                         'extended', 'long_running', 'batch_manual']:
            if median_hours <= thresholds.get(category, float('inf')):
                return category
        return 'batch_manual'

    def _get_category_label(self, category: str) -> str:
        """Get human-readable label for category"""
        labels = self._get_config('PROCESS_CATEGORY_LABELS', {
            'ultra_fast': 'Ultra Fast (<5s)',
            'very_fast': 'Very Fast (5-30s)',
            'fast_background': 'Fast Background (<6m)',
            'standard': 'Standard (6m-30m)',
            'extended': 'Extended (30m-4h)',
            'long_running': 'Long Running (4h-24h)',
            'batch_manual': 'Batch / Manual (24h+)'
        })
        return labels.get(category, category)

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float, handling None and Decimal"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _compute_cv(self, std: float, mean: float) -> Optional[float]:
        """Calculate coefficient of variation"""
        if mean is None or mean == 0:
            return None
        return std / mean

    def _classify_stability(self, cv: Optional[float]) -> str:
        """Classify process stability based on CV"""
        if cv is None:
            return 'unknown'

        stable_threshold = self._get_config('STABILITY_CV_STABLE_THRESHOLD', 0.3)
        moderate_threshold = self._get_config('STABILITY_CV_MODERATE_THRESHOLD', 1.0)

        if cv < stable_threshold:
            return 'stable'
        elif cv < moderate_threshold:
            return 'moderate'
        else:
            return 'variable'

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

    def _categorize_incident_root_cause(self, incident_type: str, error_message: str) -> str:
        """Categorize incident root cause based on type and message"""
        incident_type_lower = incident_type.lower() if incident_type else ''
        error_msg_lower = error_message.lower() if error_message else ''

        # Database/Connection issues
        if any(keyword in error_msg_lower for keyword in ['connection', 'timeout', 'database', 'sql', 'deadlock', 'jdbc']):
            return 'Database'

        # Network/External Service issues
        if any(keyword in error_msg_lower for keyword in ['network', 'unreachable', 'refused', 'http', 'rest', 'api', 'socket']):
            return 'External Service'

        # Business Logic/Validation
        if any(keyword in error_msg_lower for keyword in ['validation', 'invalid', 'null', 'constraint', 'business', 'required']):
            return 'Business Logic'

        # Configuration issues
        if any(keyword in error_msg_lower for keyword in ['config', 'property', 'missing', 'not found', 'undefined']):
            return 'Configuration'

        # Resource/Capacity issues
        if any(keyword in error_msg_lower for keyword in ['memory', 'heap', 'resource', 'capacity', 'thread', 'pool']):
            return 'Resource'

        # Authorization/Security
        if any(keyword in error_msg_lower for keyword in ['auth', 'permission', 'forbidden', 'unauthorized', 'security']):
            return 'Authorization'

        # Timeout issues
        if any(keyword in incident_type_lower for keyword in ['timeout']) or 'timeout' in error_msg_lower:
            return 'Timeout'

        # Job/Async failures
        if 'job' in incident_type_lower or 'async' in error_msg_lower:
            return 'Job Execution'

        # Default categorization
        if incident_type_lower:
            return incident_type.split('_')[0].title()

        return 'Other'

    def _get_anomaly_recommendation(self, severity: str, anomaly_types: list,
                                     deviation_pct: float, z_score: float,
                                     mean_seconds: float, stability: str, category: str) -> str:
        """Generate actionable recommendation based on anomaly characteristics"""
        if severity == 'critical':
            if 'extreme_duration' in anomaly_types:
                return "Immediate action required: Check for stuck processes, database locks, or external service timeouts"
            return "Critical performance degradation detected. Review recent deployments and infrastructure changes"

        if severity == 'high':
            if mean_seconds > 7200:  # > 2 hours
                return "Consider implementing timeout controls and async processing for long-running tasks"
            if abs(deviation_pct) > 100:
                return "Performance doubled from baseline. Investigate recent code changes or data volume increases"
            return "Significant performance degradation. Review bottleneck analysis for optimization targets"

        if severity == 'medium':
            if stability == 'variable':
                return "High variability detected. Analyze input data patterns and implement consistent resource allocation"
            if 'slow_average' in anomaly_types:
                return "Average duration increasing. Monitor resource utilization and consider scaling"
            return "Performance deviation detected. Review activity execution times and database query performance"

        # Low severity
        if abs(z_score) > 1.5:
            return "Monitor trend. If deviation persists, investigate potential infrastructure changes"
        return "Minor variation within acceptable range. Continue monitoring"

    def _generate_anomaly_reason(self, anomaly_types: list, deviation_pct: float,
                                  z_score: float, mean_seconds: float, recent_seconds: float) -> str:
        """Generate human-readable reason for anomaly detection"""
        reasons = []

        if 'statistical_deviation' in anomaly_types:
            direction = "slower" if deviation_pct > 0 else "faster"
            reasons.append(f"Running {abs(deviation_pct):.0f}% {direction} than historical baseline (z-score: {z_score:.1f})")

        if 'extreme_duration' in anomaly_types:
            if mean_seconds > 86400:  # > 1 day
                reasons.append(f"Extremely long duration: {mean_seconds/3600:.1f} hours average")
            elif mean_seconds > 7200:  # > 2 hours
                reasons.append(f"Very long duration: {mean_seconds/60:.0f} minutes average")
            else:
                reasons.append(f"Long duration: {mean_seconds:.0f} seconds average")

        if 'slow_average' in anomaly_types:
            reasons.append(f"Average execution time ({recent_seconds:.0f}s) exceeds performance threshold")

        if not reasons:
            reasons.append(f"Atypical execution pattern detected")

        return " | ".join(reasons)

    def _get_business_critical_process_keys(self, lookback_days=None):
        """
        Get list of business-critical process keys (excluding ultra_fast)
        Used to filter queries for better performance
        """
        try:
            categories_data = self.get_process_categories(lookback_days=lookback_days)
            categories = categories_data.get('categories', {})

            business_critical = [
                proc_key for proc_key, data in categories.items()
                if data.get('is_business_critical', False)
            ]

            return business_critical
        except Exception as e:
            logger.error(f"Error getting business critical processes: {e}")
            return []

    # =========================================================================
    # CLUSTER HEALTH SCORING
    # =========================================================================

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
            heap_score = max(0, 100 - heap_pct * 1.5)

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

        # Latency score using config thresholds
        excellent = self._get_config('DB_LATENCY_EXCELLENT_MS', 10)
        good = self._get_config('DB_LATENCY_GOOD_MS', 50)
        fair = self._get_config('DB_LATENCY_FAIR_MS', 100)
        poor = self._get_config('DB_LATENCY_POOR_MS', 500)

        if latency < excellent:
            latency_score = 100
        elif latency < good:
            latency_score = 80
        elif latency < fair:
            latency_score = 60
        elif latency < poor:
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

        # Use config thresholds
        excellent = self._get_config('INCIDENT_RATE_EXCELLENT_PCT', 1.0)
        good = self._get_config('INCIDENT_RATE_GOOD_PCT', 5.0)
        fair = self._get_config('INCIDENT_RATE_FAIR_PCT', 10.0)

        if incident_rate == 0:
            return 100
        elif incident_rate < excellent:
            return 90
        elif incident_rate < good:
            return 70
        elif incident_rate < fair:
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

        # Use config thresholds
        jvm_threshold = self._get_config('HEALTH_JVM_WARNING_THRESHOLD', 70)
        db_threshold = self._get_config('HEALTH_DB_WARNING_THRESHOLD', 70)
        incidents_threshold = self._get_config('HEALTH_INCIDENTS_WARNING_THRESHOLD', 80)
        jobs_threshold = self._get_config('HEALTH_JOBS_WARNING_THRESHOLD', 90)

        factors = []
        if breakdown.get('jvm', 100) < jvm_threshold:
            factors.append('JVM under pressure')
        if breakdown.get('db', 100) < db_threshold:
            factors.append('Database performance issues')
        if breakdown.get('incidents', 100) < incidents_threshold:
            factors.append('High incident rate')
        if breakdown.get('jobs', 100) < jobs_threshold:
            factors.append('Job failures detected')
        if breakdown.get('cluster', 100) < 100:
            factors.append('Node(s) down')

        return ', '.join(factors) if factors else 'All systems healthy'

    # =========================================================================
    # PROCESS CATEGORIZATION
    # =========================================================================

    def get_process_categories(self, lookback_days=None):
        """
        Categorize all processes by duration (Phase 1 of professional analysis)
        Returns process categories for intelligent filtering
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_CAPACITY_TRAINING_DAYS', 90)

            min_instances = self._get_config('AI_MIN_DATA', 10)

            query = f"""
                SELECT 
                    proc_def_key_,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as median_duration_s,
                    COUNT(*) as instance_count,
                    AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_))) as avg_duration_s,
                    STDDEV(EXTRACT(EPOCH FROM (end_time_ - start_time_))) as std_duration_s,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as p95_duration_s,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as p99_duration_s
                FROM act_hi_procinst
                WHERE start_time_ > NOW() - INTERVAL '{lookback_days} days'
                  AND end_time_ IS NOT NULL
                  AND end_time_ > start_time_
                GROUP BY proc_def_key_
                HAVING COUNT(*) >= {min_instances}
                ORDER BY median_duration_s
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Categorizing processes"
            )

            if not results:
                return {
                    'categories': {},
                    'category_counts': {},
                    'total_processes': 0,
                    'message': 'No process data available for categorization'
                }

            process_categories = {}
            category_counts = {}

            # Get analyze categories from config
            analyze_categories = self._get_config('ANALYZE_CATEGORIES',
                ['very_fast', 'fast_background', 'standard', 'extended', 'long_running', 'batch_manual'])

            for row in results:
                proc_key = row['proc_def_key_']
                median_s = self._safe_float(row['median_duration_s']) or 0
                median_hours = median_s / 3600.0

                avg_s = self._safe_float(row['avg_duration_s']) or 0
                std_s = self._safe_float(row['std_duration_s']) or 0
                p95_s = self._safe_float(row['p95_duration_s']) or 0
                p99_s = self._safe_float(row['p99_duration_s']) or 0

                # Calculate CV and classify stability
                cv = self._compute_cv(std_s, avg_s)
                stability = self._classify_stability(cv)

                # Classify into category
                category = self._classify_process_category(median_hours)

                # Calculate P95/Median ratio for variability detection
                p95_median_ratio = (p95_s / median_s) if median_s > 0 else 0

                process_categories[proc_key] = {
                    'category': category,
                    'category_label': self._get_category_label(category),
                    'median_seconds': round(median_s, 2),
                    'median_hours': round(median_hours, 4),
                    'avg_seconds': round(avg_s, 2),
                    'std_seconds': round(std_s, 2),
                    'p95_seconds': round(p95_s, 2),
                    'p99_seconds': round(p99_s, 2),
                    'p95_median_ratio': round(p95_median_ratio, 2),
                    'cv': round(cv, 4) if cv else None,
                    'stability': stability,
                    'instance_count': int(row['instance_count']),
                    'is_business_critical': category in analyze_categories
                }

                category_counts[category] = category_counts.get(category, 0) + 1

            return {
                'categories': process_categories,
                'category_counts': category_counts,
                'total_processes': len(process_categories),
                'analysis_window_days': lookback_days,
                'message': f'Categorized {len(process_categories)} processes into {len(category_counts)} categories'
            }

        except Exception as e:
            logger.error(f"Error categorizing processes: {e}")
            return {
                'categories': {},
                'category_counts': {},
                'total_processes': 0,
                'error': str(e)
            }

    # =========================================================================
    # ANOMALY DETECTION (ENHANCED)
    # =========================================================================

    def detect_process_anomalies(self, lookback_days=None):
        """
        Detect anomalies in process execution times using historical data
        Enhanced with process categorization and IQR-based outlier detection
        """
        try:
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

            # Get process categories for context
            categories_data = self.get_process_categories(lookback_days=lookback_days * 2)
            process_categories = categories_data.get('categories', {})

            # Get business-critical process keys for filtering
            business_critical_keys = self._get_business_critical_process_keys(lookback_days * 2)

            if business_critical_keys:
                # Build IN clause for filtering (limit to 500 to avoid SQL issues)
                keys_str = "', '".join(business_critical_keys[:500])
                proc_filter = f"AND proc_def_key_ IN ('{keys_str}')"
            else:
                proc_filter = ""

            # Get historical process execution data
            query = f"""
                SELECT 
                    id_ as instance_id,
                    proc_def_key_,
                    business_key_,
                    EXTRACT(EPOCH FROM (end_time_ - start_time_)) * 1000 as duration_ms,
                    start_time_,
                    end_time_
                FROM act_hi_procinst
                WHERE end_time_ IS NOT NULL
                AND start_time_ IS NOT NULL
                AND end_time_ > start_time_
                AND end_time_ > NOW() - INTERVAL '{lookback_days} days'
                {proc_filter}
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
            process_instances = defaultdict(list)  # Store full instance data
            for row in results:
                duration = float(row['duration_ms'])
                process_groups[row['proc_def_key_']].append(duration)
                process_instances[row['proc_def_key_']].append({
                    'instance_id': row['instance_id'],
                    'business_key': row.get('business_key_') or 'N/A',
                    'duration_ms': duration,
                    'start_time': row['start_time_'],
                    'end_time': row['end_time_']
                })

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

                    zscore_high = self._get_config('ANOMALY_ZSCORE_HIGH_THRESHOLD', 3.0)
                    zscore_medium = self._get_config('ANOMALY_ZSCORE_MEDIUM_THRESHOLD', 2.0)

                    if z_score > zscore_threshold:
                        detected = True
                        anomaly_types.append('statistical_deviation')
                        if z_score > zscore_high:
                            severity = 'high'
                        elif z_score > zscore_medium:
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
                    median_ms = np.median(durations)
                    p95_ms = np.percentile(durations, 95)
                    p95_ratio = (p95_ms / median_ms) if median_ms > 0 else 0
                    variability_score = round(p95_ratio, 2)

                    # Get category info if available
                    category_info = process_categories.get(proc_key, {})
                    category = category_info.get('category', 'unknown')
                    category_label = category_info.get('category_label', 'Unknown')
                    cv = category_info.get('cv')
                    stability = category_info.get('stability', 'unknown')

                    # Generate contextual recommendation
                    recommendation = self._get_anomaly_recommendation(
                        severity, anomaly_types, deviation_pct, z_score,
                        mean_seconds, stability, category
                    )

                    # Generate reason text
                    reason = self._generate_anomaly_reason(
                        anomaly_types, deviation_pct, z_score, mean_seconds, recent_mean / 1000
                    )

                    # Get sample instances - prioritize slowest ones
                    instances = process_instances.get(proc_key, [])
                    # Sort by duration descending and take top 5
                    slowest_instances = sorted(instances, key=lambda x: x['duration_ms'], reverse=True)[:5]
                    sample_instances = [{
                        'instance_id': inst['instance_id'],
                        'business_key': inst['business_key'],
                        'duration_ms': round(inst['duration_ms'], 2),
                        'start_time': inst['start_time'].isoformat() if inst['start_time'] else None,
                        'end_time': inst['end_time'].isoformat() if inst['end_time'] else None,
                        'is_slow': bool(inst['duration_ms'] > p95_ms)  # Convert numpy bool to Python bool
                    } for inst in slowest_instances]

                    anomalies.append({
                        'process_key': proc_key,
                        'category': category,
                        'category_label': category_label,
                        'current_avg_ms': round(float(recent_mean), 2),
                        'expected_avg_ms': round(float(mean_duration), 2),
                        'max_duration_ms': round(float(max_duration), 2),
                        'deviation_pct': round(float(deviation_pct), 1),
                        'z_score': round(float(z_score), 2),
                        'severity': severity,
                        'anomaly_types': anomaly_types,
                        'instances_analyzed': int(len(durations)),
                        'median_ms': round(float(median_ms), 2),
                        'p95_ms': round(float(p95_ms), 2),
                        'p95_ratio': round(float(p95_ratio), 2),
                        'variability_score': float(variability_score) if variability_score else None,
                        'cv': float(cv) if cv else None,
                        'stability': stability,
                        'recommendation': recommendation,
                        'reason': reason,
                        'sample_instances': sample_instances
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

    # =========================================================================
    # ACTIVITY BOTTLENECK ANALYSIS
    # =========================================================================

    def analyze_activity_bottlenecks(self, lookback_days=None, limit=30):
        """
        Analyze slowest activities across business-critical processes
        Returns top bottlenecks with execution metrics and sample instances
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            # Get business-critical process keys for filtering
            business_keys = self._get_business_critical_process_keys(lookback_days)

            if not business_keys:
                logger.warning("No business-critical processes found for activity bottleneck analysis")
                return []

            # Create filter for business-critical processes only
            process_filter = "', '".join(business_keys)

            query = f"""
                SELECT 
                    ai.proc_def_key_,
                    ai.act_id_,
                    ai.act_name_,
                    ai.act_type_,
                    COUNT(*) as execution_count,
                    AVG(EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as avg_duration_s,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as p50_duration_s,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as p95_duration_s,
                    STDDEV(EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as stddev_duration_s,
                    MAX(EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as max_duration_s,
                    MIN(ai.proc_inst_id_) as sample_instance_id
                FROM act_hi_actinst ai
                WHERE ai.start_time_ > NOW() - INTERVAL '{lookback_days} days'
                  AND ai.end_time_ IS NOT NULL
                  AND ai.proc_def_key_ IN ('{process_filter}')
                  AND ai.act_type_ IN ('serviceTask', 'sendTask', 'businessRuleTask', 'scriptTask', 'userTask')
                GROUP BY ai.proc_def_key_, ai.act_id_, ai.act_name_, ai.act_type_
                HAVING COUNT(*) >= 5
                ORDER BY avg_duration_s DESC
                LIMIT {limit}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Analyzing activity bottlenecks"
            )

            if not results:
                return []

            activities = []
            for row in results:
                avg_s = self._safe_float(row.get('avg_duration_s')) or 0
                stddev_s = self._safe_float(row.get('stddev_duration_s')) or 0
                p50_s = self._safe_float(row.get('p50_duration_s')) or 0
                p95_s = self._safe_float(row.get('p95_duration_s')) or 0
                max_s = self._safe_float(row.get('max_duration_s')) or 0
                count = int(row.get('execution_count', 0))

                cv = self._compute_cv(stddev_s, avg_s)
                stability = self._classify_stability(cv)

                # Calculate total time spent (hours) for optimization impact
                total_hours = (avg_s * count) / 3600

                # Calculate potential savings (10% improvement)
                potential_savings_hours = total_hours * 0.1

                # Get business key for sample instance
                sample_instance_id = row.get('sample_instance_id')
                business_key = None
                if sample_instance_id:
                    bkey_query = f"""
                        SELECT business_key_
                        FROM act_hi_procinst
                        WHERE id_ = '{sample_instance_id}'
                        LIMIT 1
                    """
                    bkey_result = safe_execute(
                        lambda: execute_query(bkey_query),
                        default_value=[],
                        context="Fetching business key for sample"
                    )
                    if bkey_result and bkey_result[0].get('business_key_'):
                        business_key = bkey_result[0]['business_key_']

                activities.append({
                    'process_key': row.get('proc_def_key_'),
                    'activity_id': row.get('act_id_'),
                    'activity_name': row.get('act_name_') or row.get('act_id_'),
                    'activity_type': row.get('act_type_'),
                    'execution_count': count,
                    'avg_duration_s': avg_s,
                    'median_duration_s': p50_s,
                    'p95_duration_s': p95_s,
                    'max_duration_s': max_s,
                    'cv': cv,
                    'stability': stability,
                    'total_time_hours': round(total_hours, 2),
                    'potential_savings_hours': round(potential_savings_hours, 2),
                    'sample_instance_id': sample_instance_id,
                    'sample_business_key': business_key
                })

            logger.info(f"Analyzed {len(activities)} activity bottlenecks from {len(business_keys)} business-critical processes")
            return activities

        except Exception as e:
            logger.error(f"Error analyzing activity bottlenecks: {e}")
            return []

    # =========================================================================
    # INCIDENT PATTERN ANALYSIS (ENHANCED)
    # =========================================================================

    def analyze_incident_patterns(self, lookback_days=None):
        """
        Cluster similar incidents using text analysis
        Enhanced with business keys, status tracking, and sample instances
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            max_incidents = self._get_config('AI_MAX_INCIDENTS', 1000)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)
            sample_limit = self._get_config('INCIDENT_PATTERN_SAMPLE_LIMIT', 5)

            # First try historical incidents with full context
            query = f"""
                SELECT 
                    inc.incident_type_,
                    inc.incident_msg_,
                    inc.proc_def_key_,
                    inc.activity_id_,
                    inc.create_time_,
                    inc.end_time_,
                    inc.proc_inst_id_,
                    pi.business_key_,
                    CASE WHEN inc.end_time_ IS NULL THEN 'open' ELSE 'resolved' END as status,
                    EXTRACT(EPOCH FROM (COALESCE(inc.end_time_, NOW()) - inc.create_time_)) as duration_seconds
                FROM act_hi_incident inc
                LEFT JOIN act_hi_procinst pi ON inc.proc_inst_id_ = pi.id_
                WHERE inc.create_time_ > NOW() - INTERVAL '{lookback_days} days'
                ORDER BY inc.create_time_ DESC
                LIMIT {max_incidents}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Analyzing incident patterns"
            )

            # If no historical incidents, try runtime job exceptions
            if not results or len(results) < 1:
                logger.info("No historical incidents found, analyzing runtime job exceptions")

                query_fallback = f"""
                    SELECT 
                        COALESCE(jl.job_def_type_, 'unknown') as incident_type_,
                        jl.job_exception_msg_ as incident_msg_,
                        jl.process_def_key_ as proc_def_key_,
                        jl.act_id_ as activity_id_,
                        jl.timestamp_ as create_time_,
                        NULL as end_time_,
                        jl.process_instance_id_ as proc_inst_id_,
                        pi.business_key_,
                        'open' as status,
                        0 as duration_seconds
                    FROM act_hi_job_log jl
                    LEFT JOIN act_hi_procinst pi ON jl.process_instance_id_ = pi.id_
                    WHERE jl.timestamp_ > NOW() - INTERVAL '{lookback_days} days'
                      AND jl.job_exception_msg_ IS NOT NULL
                    ORDER BY jl.timestamp_ DESC
                    LIMIT {max_incidents}
                """

                fallback_results = safe_execute(
                    lambda: execute_query(query_fallback),
                    default_value=[],
                    context="Analyzing job exceptions as incidents"
                )

                if fallback_results and len(fallback_results) > 0:
                    results = fallback_results
                    data_source = 'runtime_jobs'
                else:
                    # Try activity failures as last fallback
                    logger.info("No job exceptions found, analyzing activity failures")

                    query_activities = f"""
                        SELECT 
                            ai.act_type_ as incident_type_,
                            'Activity did not complete' as incident_msg_,
                            ai.proc_def_key_,
                            ai.act_id_ as activity_id_,
                            ai.start_time_ as create_time_,
                            ai.end_time_,
                            ai.proc_inst_id_,
                            pi.business_key_,
                            CASE WHEN ai.end_time_ IS NULL THEN 'open' ELSE 'resolved' END as status,
                            EXTRACT(EPOCH FROM (COALESCE(ai.end_time_, NOW()) - ai.start_time_)) as duration_seconds
                        FROM act_hi_actinst ai
                        LEFT JOIN act_hi_procinst pi ON ai.proc_inst_id_ = pi.id_
                        WHERE ai.start_time_ > NOW() - INTERVAL '{lookback_days} days'
                          AND ai.end_time_ IS NULL
                          AND ai.act_type_ IN ('serviceTask', 'sendTask', 'businessRuleTask', 'scriptTask', 'userTask')
                        ORDER BY ai.start_time_ DESC
                        LIMIT {max_incidents}
                    """

                    activity_results = safe_execute(
                        lambda: execute_query(query_activities),
                        default_value=[],
                        context="Analyzing activity failures"
                    )

                    if activity_results:
                        results = activity_results
                        data_source = 'activity_failures'
                    else:
                        return {
                            'patterns': [],
                            'total_incidents': 0,
                            'unique_patterns': 0,
                            'analysis_window_days': lookback_days,
                            'message': f'✓ No incidents found in last {lookback_days} days - System is healthy!',
                            'health_status': 'excellent',
                            'data_source': 'none'
                        }
            else:
                data_source = 'historical'

            # Process incidents
            pattern_groups = defaultdict(lambda: {
                'incidents': [],
                'open_count': 0,
                'resolved_count': 0,
                'total_duration': 0,
                'sample_instances': []
            })

            for row in results:
                incident_type = row['incident_type_'] if row.get('incident_type_') else 'Unknown Type'
                incident_msg = row['incident_msg_'][:100] if row.get('incident_msg_') else 'No error message'
                key = f"{incident_type}:::{incident_msg}"

                pattern_groups[key]['incidents'].append(row)

                # Track status
                status = row.get('status', 'unknown')
                if status == 'open':
                    pattern_groups[key]['open_count'] += 1
                elif status == 'resolved':
                    pattern_groups[key]['resolved_count'] += 1

                # Track duration
                duration = self._safe_float(row.get('duration_seconds'))
                if duration:
                    pattern_groups[key]['total_duration'] += duration

                # Store sample instances (max configured per pattern)
                if len(pattern_groups[key]['sample_instances']) < sample_limit:
                    pattern_groups[key]['sample_instances'].append({
                        'instance_id': row.get('proc_inst_id_', 'N/A'),
                        'business_key': row.get('business_key_', 'N/A'),
                        'status': status,
                        'create_time': row['create_time'].isoformat() if row.get('create_time') else None
                    })

            patterns = []
            for key, data in pattern_groups.items():
                parts = key.split(':::', 1)
                incident_type = parts[0] if len(parts) > 0 else 'Unknown'
                msg = parts[1] if len(parts) > 1 else 'No message'

                incidents = data['incidents']
                affected_processes = list(set([i['proc_def_key_'] for i in incidents if i.get('proc_def_key_')]))
                affected_activities = list(set([i['activity_id_'] for i in incidents if i.get('activity_id_')]))

                timestamps = [i['create_time'] for i in incidents if i.get('create_time')]
                first_seen = min(timestamps).isoformat() if timestamps else None
                last_seen = max(timestamps).isoformat() if timestamps else None

                total_count = len(incidents)
                avg_duration = (data['total_duration'] / total_count) if total_count > 0 else 0

                patterns.append({
                    'incident_type': incident_type,
                    'error_message': msg,
                    'occurrence_count': total_count,
                    'open_count': data['open_count'],
                    'resolved_count': data['resolved_count'],
                    'avg_duration_hours': round(avg_duration / 3600, 2) if avg_duration > 0 else 0,
                    'affected_processes': affected_processes[:5],
                    'affected_activities': affected_activities[:5],
                    'first_seen': first_seen,
                    'last_seen': last_seen,
                    'frequency_per_day': round(total_count / max(lookback_days, 1), 2),
                    'sample_instances': data['sample_instances'],
                    'source': data_source,
                    'health_status': 'critical' if data['open_count'] > total_count * 0.5 else 'degraded' if data['open_count'] > 0 else 'healthy'
                })

            patterns.sort(key=lambda x: x['occurrence_count'], reverse=True)

            # Categorize root causes
            root_cause_categories = defaultdict(int)
            for pattern in patterns:
                root_cause = self._categorize_incident_root_cause(pattern['incident_type'], pattern['error_message'])
                pattern['root_cause'] = root_cause
                root_cause_categories[root_cause] += pattern['occurrence_count']

            return {
                'patterns': patterns[:max_results],
                'total_incidents': len(results),
                'unique_patterns': len(patterns),
                'total_open': sum(p['open_count'] for p in patterns),
                'total_resolved': sum(p['resolved_count'] for p in patterns),
                'root_cause_categories': dict(root_cause_categories),
                'analysis_window_days': lookback_days,
                'message': f'Found {len(patterns)} unique incident patterns from {len(results)} incidents',
                'data_source': data_source
            }

        except Exception as e:
            logger.error(f"Error analyzing incidents: {e}")
            return {
                'patterns': [],
                'total_incidents': 0,
                'error': str(e),
                'data_source': 'error'
            }

    # =========================================================================
    # VERSION PERFORMANCE ANALYSIS
    # =========================================================================

    def analyze_version_performance(self, lookback_days=None, include_all_versions=False):
        """
        Analyze performance changes between process versions
        Detects regressions and improvements across deployments

        Args:
            lookback_days: Number of days to analyze
            include_all_versions: If True, returns complete version history per process
                                  If False, returns only latest 2 versions comparison (legacy behavior)
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_CAPACITY_TRAINING_DAYS', 90)

            min_instances = self._get_config('AI_MIN_DATA', 10)
            regression_threshold = self._get_config('VERSION_REGRESSION_THRESHOLD_PCT', 20.0)
            high_threshold = self._get_config('VERSION_REGRESSION_HIGH_PCT', 50.0)
            critical_threshold = self._get_config('VERSION_REGRESSION_CRITICAL_PCT', 100.0)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            # Get business-critical process keys for filtering
            business_keys = self._get_business_critical_process_keys(lookback_days)

            process_filter = ""
            if business_keys:
                process_list = "', '".join(business_keys)
                process_filter = f"AND proc_def_key_ IN ('{process_list}')"

            query = f"""
                WITH version_data AS (
                    SELECT 
                        pi.proc_def_key_,
                        SUBSTRING(pi.proc_def_id_ FROM ':([0-9]+):') as version,
                        EXTRACT(EPOCH FROM (pi.end_time_ - pi.start_time_)) as duration_s,
                        pi.start_time_,
                        pi.id_ as instance_id,
                        pi.business_key_
                    FROM act_hi_procinst pi
                    WHERE pi.start_time_ > NOW() - INTERVAL '{lookback_days} days'
                      AND pi.end_time_ IS NOT NULL
                      AND pi.end_time_ > pi.start_time_
                      {process_filter}
                )
                SELECT 
                    proc_def_key_,
                    version,
                    COUNT(*) as instance_count,
                    AVG(duration_s) as avg_duration_s,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_s) as p50_duration_s,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_s) as p95_duration_s,
                    STDDEV(duration_s) as std_duration_s,
                    MIN(start_time_) as first_used,
                    MAX(start_time_) as last_used,
                    MIN(instance_id) as sample_instance_id,
                    (ARRAY_AGG(business_key_ ORDER BY start_time_ DESC) FILTER (WHERE business_key_ IS NOT NULL))[1] as sample_business_key
                FROM version_data
                WHERE version IS NOT NULL
                GROUP BY proc_def_key_, version
                HAVING COUNT(*) >= {min_instances}
                ORDER BY proc_def_key_, CAST(version AS INTEGER) DESC
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Analyzing version performance"
            )

            if not results:
                return {
                    'version_comparisons': [],
                    'version_history': [],
                    'regressions': [],
                    'improvements': [],
                    'total_processes': 0,
                    'message': 'No version data available'
                }

            # Group by process
            by_process = defaultdict(list)
            for row in results:
                by_process[row['proc_def_key_']].append(row)

            version_comparisons = []
            regressions = []
            improvements = []

            for proc_key, versions in by_process.items():
                if len(versions) < 2:
                    continue

                # Sort by version number (descending)
                try:
                    versions_sorted = sorted(versions, key=lambda x: int(x['version']), reverse=True)
                except:
                    versions_sorted = versions

                # Compare latest vs previous
                latest = versions_sorted[0]
                previous = versions_sorted[1]

                latest_avg = self._safe_float(latest['avg_duration_s']) or 0
                previous_avg = self._safe_float(previous['avg_duration_s']) or 0
                latest_p95 = self._safe_float(latest['p95_duration_s']) or 0
                previous_p95 = self._safe_float(previous['p95_duration_s']) or 0

                if previous_avg > 0:
                    change_pct = ((latest_avg - previous_avg) / previous_avg) * 100
                    p95_change_pct = ((latest_p95 - previous_p95) / previous_p95) * 100 if previous_p95 > 0 else 0

                    # Determine severity using config thresholds
                    if change_pct > critical_threshold:
                        severity = 'critical'
                    elif change_pct > high_threshold:
                        severity = 'high'
                    elif abs(change_pct) > regression_threshold:
                        severity = 'medium'
                    else:
                        severity = 'low'

                    comparison = {
                        'process_key': proc_key,
                        'latest_version': latest['version'],
                        'previous_version': previous['version'],
                        'latest_avg_s': round(latest_avg, 2),
                        'previous_avg_s': round(previous_avg, 2),
                        'latest_p95_s': round(latest_p95, 2),
                        'previous_p95_s': round(previous_p95, 2),
                        'change_pct': round(change_pct, 1),
                        'p95_change_pct': round(p95_change_pct, 1),
                        'latest_count': int(latest['instance_count']),
                        'previous_count': int(previous['instance_count']),
                        'first_used': latest['first_used'].isoformat() if latest.get('first_used') else None,
                        'last_used': latest['last_used'].isoformat() if latest.get('last_used') else None,
                        'direction': 'regression' if change_pct > regression_threshold else 'improvement' if change_pct < -regression_threshold else 'stable',
                        'severity': severity,
                        'sample_instance_id': latest.get('sample_instance_id'),
                        'sample_business_key': latest.get('sample_business_key'),
                        'all_versions_count': len(versions_sorted),
                        'all_versions': [
                            {
                                'version': v['version'],
                                'avg_s': round(self._safe_float(v['avg_duration_s']) or 0, 2),
                                'p95_s': round(self._safe_float(v['p95_duration_s']) or 0, 2),
                                'count': int(v['instance_count']),
                                'first_used': v['first_used'].isoformat() if v.get('first_used') else None,
                                'last_used': v['last_used'].isoformat() if v.get('last_used') else None,
                                'sample_instance_id': v.get('sample_instance_id'),
                                'sample_business_key': v.get('sample_business_key')
                            } for v in versions_sorted
                        ] if include_all_versions else []
                    }

                    version_comparisons.append(comparison)

                    # Track regressions and improvements
                    if change_pct > regression_threshold:
                        regressions.append({
                            **comparison,
                            'recommendation': f'ROLLBACK: Version {latest["version"]} is {change_pct:.1f}% slower than v{previous["version"]} - Consider reverting'
                        })
                    elif change_pct < -regression_threshold:
                        improvements.append({
                            **comparison,
                            'note': f'IMPROVEMENT: Version {latest["version"]} is {abs(change_pct):.1f}% faster than v{previous["version"]}'
                        })

            # Sort by severity
            regressions.sort(key=lambda x: x['change_pct'], reverse=True)
            improvements.sort(key=lambda x: x['change_pct'])

            result = {
                'version_comparisons': version_comparisons,
                'regressions': regressions[:max_results],
                'improvements': improvements[:max_results],
                'total_processes': len(by_process),
                'processes_with_versions': len(version_comparisons),
                'regression_count': len(regressions),
                'improvement_count': len(improvements),
                'analysis_window_days': lookback_days,
                'message': f'Analyzed {len(version_comparisons)} processes with version changes'
            }

            # Add flag to indicate if complete history is included
            if include_all_versions:
                result['includes_complete_history'] = True

            return result

        except Exception as e:
            logger.error(f"Error analyzing version performance: {e}")
            return {
                'version_comparisons': [],
                'regressions': [],
                'improvements': [],
                'total_processes': 0,
                'error': str(e)
            }

    # =========================================================================
    # EXTREME VARIABILITY DETECTION
    # =========================================================================

    def analyze_extreme_variability(self, process_categories=None):
        """
        Detect processes with extreme P95/Median ratios (dangerously unpredictable)
        ENHANCED: Now includes sample slow instances with business keys
        """
        try:
            if process_categories is None:
                cat_result = self.get_process_categories()
                process_categories = cat_result.get('categories', {})

            if not process_categories:
                return {
                    'extreme_processes': [],
                    'total_analyzed': 0,
                    'message': 'No process category data available'
                }

            extreme_processes = []
            lookback_days = self._get_config('AI_CAPACITY_TRAINING_DAYS', 90)

            # Use config thresholds
            extreme_threshold = self._get_config('EXTREME_VARIABILITY_RATIO_EXTREME', 100.0)
            high_threshold = self._get_config('EXTREME_VARIABILITY_RATIO_HIGH', 50.0)
            medium_threshold = self._get_config('EXTREME_VARIABILITY_RATIO_MEDIUM', 20.0)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)
            sample_limit = self._get_config('INCIDENT_PATTERN_SAMPLE_LIMIT', 5)

            for proc_key, data in process_categories.items():
                p95_median_ratio = data.get('p95_median_ratio', 0)
                median_s = data.get('median_seconds', 0)
                p95_s = data.get('p95_seconds', 0)
                category = data.get('category', 'unknown')

                # Flag processes where P95 is significantly higher than median
                if p95_median_ratio > medium_threshold:
                    if p95_median_ratio > extreme_threshold:
                        severity = 'extreme'
                    elif p95_median_ratio > high_threshold:
                        severity = 'high'
                    else:
                        severity = 'medium'

                    # ENHANCEMENT: Get sample slow instances
                    sample_query = f"""
                        SELECT 
                            id_ as instance_id,
                            business_key_,
                            EXTRACT(EPOCH FROM (end_time_ - start_time_)) as duration_s,
                            start_time_,
                            end_time_
                        FROM act_hi_procinst
                        WHERE proc_def_key_ = '{proc_key}'
                          AND end_time_ IS NOT NULL
                          AND start_time_ > NOW() - INTERVAL '{lookback_days} days'
                          AND EXTRACT(EPOCH FROM (end_time_ - start_time_)) > {p95_s}
                        ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_)) DESC
                        LIMIT {sample_limit}
                    """

                    sample_results = safe_execute(
                        lambda: execute_query(sample_query),
                        default_value=[],
                        context=f"Getting sample instances for {proc_key}"
                    )

                    sample_instances = [{
                        'instance_id': row['instance_id'],
                        'business_key': row.get('business_key_', 'N/A'),
                        'duration_seconds': round(float(row['duration_s']), 2),
                        'start_time': row['start_time'].isoformat() if row.get('start_time') else None,
                        'end_time': row['end_time'].isoformat() if row.get('end_time') else None,
                        'exceeds_p95': True
                    } for row in sample_results]

                    extreme_processes.append({
                        'process_key': proc_key,
                        'category': category,
                        'category_label': data.get('category_label', category),
                        'median_seconds': median_s,
                        'p95_seconds': p95_s,
                        'p95_median_ratio': p95_median_ratio,
                        'cv': data.get('cv'),
                        'stability': data.get('stability', 'unknown'),
                        'instance_count': data.get('instance_count', 0),
                        'severity': severity,
                        'message': f'P95 is {p95_median_ratio:.0f}x longer than median - HIGHLY unpredictable',
                        'recommendation': self._get_variability_recommendation(p95_median_ratio, severity),
                        'sample_slow_instances': sample_instances  # NEW
                    })

            # Sort by ratio (worst first)
            extreme_processes.sort(key=lambda x: x['p95_median_ratio'], reverse=True)

            return {
                'extreme_processes': extreme_processes[:max_results],
                'total_analyzed': len(process_categories),
                'extreme_count': len(extreme_processes),
                'message': f'Found {len(extreme_processes)} processes with extreme variability' if extreme_processes else 'No extreme variability detected'
            }

        except Exception as e:
            logger.error(f"Error analyzing extreme variability: {e}")
            return {
                'extreme_processes': [],
                'total_analyzed': 0,
                'error': str(e)
            }

    def _get_variability_recommendation(self, ratio: float, severity: str) -> str:
        """Get recommendation for processes with extreme variability"""
        if severity == 'extreme':
            return f'CRITICAL: P95 is {ratio:.0f}x median - Do NOT use for SLA-critical workflows. Investigate external dependencies, stuck subprocesses, or consider splitting into fast/slow paths.'
        elif severity == 'high':
            return f'WARNING: P95 is {ratio:.0f}x median - Implement timeout safeguards and add monitoring alerts for P99 violations.'
        else:
            return f'CAUTION: P95 is {ratio:.0f}x median - Monitor closely and investigate outlier patterns.'

    # =========================================================================
    # LOAD PATTERN ANALYSIS
    # =========================================================================

    def analyze_load_patterns(self, lookback_days=None):
        """
        Enterprise load pattern analysis: business days vs weekends, peak hours
        Critical for capacity planning and batch job scheduling
        ENHANCED: Now filters to business-critical processes only for better performance
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_CAPACITY_TRAINING_DAYS', 90)

            business_hours_start = self._get_config('BUSINESS_HOURS_START', 7)
            business_hours_end = self._get_config('BUSINESS_HOURS_END', 19)
            weekend_days = self._get_config('WEEKEND_DAYS', [0, 6])
            peak_hours_limit = self._get_config('LOAD_PATTERN_PEAK_HOURS_LIMIT', 10)

            # Get business-critical process keys for filtering
            business_keys = self._get_business_critical_process_keys(lookback_days)

            process_filter = ""
            if business_keys:
                process_list = "', '".join(business_keys)
                process_filter = f"AND proc_def_key_ IN ('{process_list}')"

            # Daily patterns (business vs weekend)
            day_query = f"""
                SELECT 
                    EXTRACT(DOW FROM start_time_) as day_of_week,
                    CASE EXTRACT(DOW FROM start_time_)
                        WHEN 0 THEN 'Sunday'
                        WHEN 1 THEN 'Monday'
                        WHEN 2 THEN 'Tuesday'
                        WHEN 3 THEN 'Wednesday'
                        WHEN 4 THEN 'Thursday'
                        WHEN 5 THEN 'Friday'
                        WHEN 6 THEN 'Saturday'
                    END as day_name,
                    COUNT(*) as total_instances,
                    AVG(EXTRACT(EPOCH FROM (COALESCE(end_time_, NOW()) - start_time_))) as avg_duration_s,
                    MAX(EXTRACT(EPOCH FROM (COALESCE(end_time_, NOW()) - start_time_))) as max_duration_s
                FROM act_hi_procinst
                WHERE start_time_ > NOW() - INTERVAL '{lookback_days} days'
                  {process_filter}
                GROUP BY EXTRACT(DOW FROM start_time_)
                ORDER BY day_of_week
            """

            day_results = safe_execute(
                lambda: execute_query(day_query),
                default_value=[],
                context="Analyzing daily load patterns"
            )

            # Hourly patterns
            hour_query = f"""
                SELECT 
                    EXTRACT(HOUR FROM start_time_) as hour_of_day,
                    COUNT(*) as hourly_instances,
                    AVG(EXTRACT(EPOCH FROM (COALESCE(end_time_, NOW()) - start_time_))) as avg_duration_s
                FROM act_hi_procinst
                WHERE start_time_ > NOW() - INTERVAL '{lookback_days} days'
                  {process_filter}
                GROUP BY EXTRACT(HOUR FROM start_time_)
                ORDER BY hourly_instances DESC"""

            hour_results = safe_execute(
                lambda: execute_query(hour_query),
                default_value=[],
                context="Analyzing hourly load patterns"
            )

            if not day_results:
                return {
                    'daily_patterns': [],
                    'hourly_patterns': [],
                    'business_summary': {},
                    'message': 'No load pattern data available'
                }

            # Process daily patterns
            daily_patterns = []
            business_totals = {'instances': 0, 'duration': 0, 'count': 0}
            weekend_totals = {'instances': 0, 'duration': 0, 'count': 0}

            for row in day_results:
                day_num = int(row['day_of_week'])
                day_name = row['day_name']
                total = int(row['total_instances'])
                avg_dur = self._safe_float(row['avg_duration_s']) or 0
                max_dur = self._safe_float(row['max_duration_s']) or 0

                is_weekend = day_num in weekend_days

                daily_patterns.append({
                    'day_of_week': day_num,
                    'day_name': day_name,
                    'is_weekend': is_weekend,
                    'total_instances': total,
                    'avg_duration_seconds': round(avg_dur, 2),
                    'max_duration_seconds': round(max_dur, 2),
                    'category': 'weekend' if is_weekend else 'business'
                })

                if is_weekend:
                    weekend_totals['instances'] += total
                    weekend_totals['duration'] += avg_dur
                    weekend_totals['count'] += 1
                else:
                    business_totals['instances'] += total
                    business_totals['duration'] += avg_dur
                    business_totals['count'] += 1

            # Calculate averages
            business_avg_duration = (business_totals['duration'] / business_totals['count']) if business_totals['count'] > 0 else 0
            weekend_avg_duration = (weekend_totals['duration'] / weekend_totals['count']) if weekend_totals['count'] > 0 else 0

            weekend_vs_business_factor = (weekend_avg_duration / business_avg_duration) if business_avg_duration > 0 else 1
            weekend_vs_business_pct = (weekend_vs_business_factor - 1) * 100

            # Process hourly patterns
            hourly_patterns = []
            peak_hours = []

            for row in hour_results:
                hour = int(row['hour_of_day'])
                count = int(row['hourly_instances'])
                avg = self._safe_float(row['avg_duration_s']) or 0

                is_business_hours = business_hours_start <= hour < business_hours_end

                hourly_patterns.append({
                    'hour': hour,
                    'instances': count,
                    'avg_duration_seconds': round(avg, 2),
                    'is_business_hours': is_business_hours,
                    'category': 'business_hours' if is_business_hours else 'after_hours'
                })

                if len(peak_hours) < peak_hours_limit:
                    peak_hours.append({
                        'hour': hour,
                        'instances': count,
                        'is_business_hours': is_business_hours
                    })

            # Business summary
            business_summary = {
                'business_days': {
                    'total_instances': business_totals['instances'],
                    'avg_duration_seconds': round(business_avg_duration, 2),
                    'days_analyzed': business_totals['count']
                },
                'weekends': {
                    'total_instances': weekend_totals['instances'],
                    'avg_duration_seconds': round(weekend_avg_duration, 2),
                    'days_analyzed': weekend_totals['count']
                },
                'weekend_vs_business': {
                    'factor': round(weekend_vs_business_factor, 2),
                    'change_pct': round(weekend_vs_business_pct, 1),
                    'direction': 'slower' if weekend_vs_business_factor > 1.1 else 'faster' if weekend_vs_business_factor < 0.9 else 'similar'
                },
                'peak_hours': peak_hours,
                'recommendations': self._get_load_pattern_recommendations(
                    business_totals, weekend_totals, peak_hours
                )
            }

            return {
                'daily_patterns': daily_patterns,
                'hourly_patterns': hourly_patterns,
                'business_summary': business_summary,
                'analysis_window_days': lookback_days,
                'message': f'Analyzed {len(daily_patterns)} days and {len(hourly_patterns)} hourly patterns'
            }

        except Exception as e:
            logger.error(f"Error analyzing load patterns: {e}")
            return {
                'daily_patterns': [],
                'hourly_patterns': [],
                'business_summary': {},
                'error': str(e)
            }

    def _get_load_pattern_recommendations(self, business_totals, weekend_totals, peak_hours):
        """Generate recommendations based on load patterns"""
        recommendations = []

        # Use config threshold
        weekend_low_threshold = self._get_config('LOAD_PATTERN_WEEKEND_LOW_THRESHOLD_PCT', 20.0)

        # Weekend utilization
        total = business_totals['instances'] + weekend_totals['instances']
        weekend_pct = (weekend_totals['instances'] / total * 100) if total > 0 else 0

        if weekend_pct < weekend_low_threshold:
            recommendations.append({
                'priority': 'medium',
                'category': 'capacity',
                'message': f'Weekend load is only {weekend_pct:.1f}% of total - Schedule batch jobs on weekends',
                'actions': ['Data cleanup', 'Report generation', 'Archive operations']
            })

        # Peak hour optimization
        if peak_hours:
            peak_hour = peak_hours[0]['hour']
            recommendations.append({
                'priority': 'high',
                'category': 'performance',
                'message': f'Peak hour is {peak_hour:02d}:00 - Optimize for peak load',
                'actions': [
                    f'Pre-warm caches before {peak_hour:02d}:00',
                    f'Scale up resources at {(peak_hour-1):02d}:00',
                    f'Delay non-critical jobs until after {(peak_hour+2):02d}:00'
                ]
            })

        return recommendations

    # =========================================================================
    # STUCK PROCESS DETECTION
    # =========================================================================

    def analyze_stuck_processes(self, lookback_days=None):
        """
        Detect stuck process instances with business keys and instance IDs
        Critical for operations teams to investigate and resolve
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            min_instances = self._get_config('AI_MIN_DATA', 5)
            p95_multiplier_critical = self._get_config('STUCK_PROCESS_P95_MULTIPLIER_CRITICAL', 3.0)
            p95_multiplier_warning = self._get_config('STUCK_PROCESS_P95_MULTIPLIER_WARNING', 2.0)
            p95_multiplier_attention = self._get_config('STUCK_PROCESS_P95_MULTIPLIER_ATTENTION', 1.5)
            fallback_threshold = self._get_config('STUCK_ACTIVITY_FALLBACK_THRESHOLD_SECONDS', 86400)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            query = f"""
                WITH running AS (
                    SELECT 
                        pi.id_ as instance_id,
                        pi.proc_def_key_,
                        pi.business_key_,
                        pi.start_time_,
                        EXTRACT(EPOCH FROM (NOW() - pi.start_time_)) as current_duration_seconds
                    FROM act_hi_procinst pi
                    WHERE pi.end_time_ IS NULL
                ), 
                base AS (
                    SELECT 
                        proc_def_key_,
                        AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_))) as avg_dur,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as p95_dur,
                        COUNT(*) as historical_count
                    FROM act_hi_procinst
                    WHERE end_time_ IS NOT NULL
                      AND start_time_ > NOW() - INTERVAL '{lookback_days * 3} days'
                    GROUP BY proc_def_key_
                    HAVING COUNT(*) >= {min_instances}
                )
                SELECT 
                    r.instance_id,
                    r.proc_def_key_,
                    r.business_key_,
                    r.start_time_,
                    r.current_duration_seconds,
                    b.avg_dur,
                    b.p95_dur,
                    b.historical_count
                FROM running r
                LEFT JOIN base b ON r.proc_def_key_ = b.proc_def_key_
                ORDER BY r.current_duration_seconds DESC
                LIMIT 1000
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Analyzing stuck processes"
            )

            if not results:
                return {
                    'stuck_processes': [],
                    'status_counts': {},
                    'total_running': 0,
                    'message': 'No running process instances found'
                }

            stuck_list = []
            status_counts = {'critical': 0, 'warning': 0, 'attention': 0, 'normal': 0}

            for row in results:
                current_dur = self._safe_float(row['current_duration_seconds']) or 0
                p95 = self._safe_float(row.get('p95_dur'))
                avg = self._safe_float(row.get('avg_dur'))

                # Determine status based on P95 comparison
                if p95 and p95 > 0:
                    if current_dur > p95 * p95_multiplier_critical:
                        status = 'critical'
                    elif current_dur > p95 * p95_multiplier_warning:
                        status = 'warning'
                    elif current_dur > (avg or p95) * p95_multiplier_attention:
                        status = 'attention'
                    else:
                        status = 'normal'
                else:
                    # No historical data - use fallback threshold
                    if current_dur > fallback_threshold:
                        status = 'attention'
                    else:
                        status = 'normal'

                status_counts[status] += 1

                # Only include critical, warning, and attention instances
                if status in ['critical', 'warning', 'attention']:
                    stuck_list.append({
                        'instance_id': row['instance_id'],
                        'process_key': row['proc_def_key_'],
                        'business_key': row.get('business_key_', 'N/A'),
                        'start_time': row['start_time'].isoformat() if row.get('start_time') else None,
                        'stuck_for_seconds': round(current_dur, 2),
                        'stuck_for_hours': round(current_dur / 3600, 2),
                        'stuck_for_days': round(current_dur / 86400, 2),
                        'expected_p95_seconds': round(p95, 2) if p95 else None,
                        'expected_p95_hours': round(p95 / 3600, 2) if p95 else None,
                        'avg_duration_seconds': round(avg, 2) if avg else None,
                        'duration_vs_p95_ratio': round(current_dur / p95, 1) if p95 and p95 > 0 else None,
                        'status': status,
                        'historical_count': int(row.get('historical_count') or 0),
                        'message': self._get_stuck_message(status, current_dur, p95)
                    })

            # Sort by duration (longest stuck first)
            stuck_list.sort(key=lambda x: x['stuck_for_seconds'], reverse=True)

            return {
                'stuck_processes': stuck_list[:max_results],
                'status_counts': status_counts,
                'total_running': len(results),
                'total_stuck': len(stuck_list),
                'analysis_window_days': lookback_days,
                'message': f'Found {len(stuck_list)} stuck instances out of {len(results)} running processes'
            }

        except Exception as e:
            logger.error(f"Error analyzing stuck processes: {e}")
            return {
                'stuck_processes': [],
                'status_counts': {},
                'total_running': 0,
                'error': str(e)
            }

    def _get_stuck_message(self, status: str, current_dur: float, p95: Optional[float]) -> str:
        """Generate human-readable message for stuck process"""
        hours = current_dur / 3600

        if status == 'critical':
            if p95:
                return f"CRITICAL: Running for {hours:.1f}h ({current_dur / p95:.1f}x longer than P95) - Immediate investigation required"
            else:
                return f"CRITICAL: Running for {hours:.1f}h with no completion in sight"
        elif status == 'warning':
            if p95:
                return f"WARNING: Running for {hours:.1f}h ({current_dur / p95:.1f}x longer than P95) - Check for stuck activities"
            else:
                return f"WARNING: Running for {hours:.1f}h - Monitor closely"
        else:  # attention
            return f"Attention: Running for {hours:.1f}h - Longer than typical but within acceptable range"

    def _get_stuck_recommendation(self, severity: str, max_running_seconds: float, stuck_count: int) -> str:
        """Generate recommendation for stuck processes"""
        hours = max_running_seconds / 3600
        
        if severity == 'critical':
            return f"CRITICAL: {stuck_count} instance(s) stuck for up to {hours:.1f}h - Terminate or investigate immediately to prevent resource exhaustion"
        elif severity == 'warning':
            return f"WARNING: {stuck_count} instance(s) running longer than expected ({hours:.1f}h max) - Review for stuck activities or external dependencies"
        else:
            return f"Monitor {stuck_count} instance(s) - Currently within acceptable range but exceeding typical duration"

    # =========================================================================
    # OUTLIER PATTERN ANALYSIS
    # =========================================================================

    def analyze_outlier_patterns(self, lookback_days=None):
        """
        IQR-based outlier detection for each process
        ENHANCED: Now includes extreme outlier tracking, sample instances, and process category filtering
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_CAPACITY_TRAINING_DAYS', 90)

            min_instances = self._get_config('AI_MIN_DATA', 10)
            iqr_normal = self._get_config('OUTLIER_IQR_MULTIPLIER_NORMAL', 1.5)
            iqr_extreme = self._get_config('OUTLIER_IQR_MULTIPLIER_EXTREME', 3.0)
            high_outlier_threshold = self._get_config('OUTLIER_HIGH_PERCENTAGE_THRESHOLD', 15.0)
            medium_outlier_threshold = self._get_config('OUTLIER_MEDIUM_PERCENTAGE_THRESHOLD', 5.0)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)
            sample_limit = self._get_config('INCIDENT_PATTERN_SAMPLE_LIMIT', 5)

            # Get business-critical process keys for filtering
            business_keys = self._get_business_critical_process_keys(lookback_days)

            process_filter = ""
            if business_keys:
                process_list = "', '".join(business_keys)
                process_filter = f"AND proc_def_key_ IN ('{process_list}')"

            # Get quartile data for business-critical processes only
            query = f"""
                SELECT 
                    proc_def_key_,
                    COUNT(*) as total_count,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as q1,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as q3,
                    AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_))) as avg_duration
                FROM act_hi_procinst
                WHERE start_time_ > NOW() - INTERVAL '{lookback_days} days'
                  AND end_time_ IS NOT NULL
                  AND end_time_ > start_time_
                  {process_filter}
                GROUP BY proc_def_key_
                HAVING COUNT(*) >= {min_instances}
            """

            results = safe_execute(
                lambda: execute_query(query),
                default_value=[],
                context="Analyzing outlier patterns"
            )

            if not results:
                return {
                    'outlier_analysis': [],
                    'high_outlier_processes': [],
                    'total_analyzed': 0,
                    'message': 'No data available for outlier analysis'
                }

            outlier_analysis = []
            high_outlier_processes = []

            for row in results:
                proc_key = row['proc_def_key_']
                q1 = self._safe_float(row['q1']) or 0
                q3 = self._safe_float(row['q3']) or 0
                avg = self._safe_float(row['avg_duration']) or 0
                total = int(row['total_count'])

                iqr = q3 - q1

                # Calculate outlier thresholds
                lower_bound = max(0, q1 - iqr_normal * iqr)
                upper_bound = q3 + iqr_normal * iqr
                extreme_upper = q3 + iqr_extreme * iqr  # ENHANCED

                # Count actual outliers AND extreme outliers
                outlier_query = f"""
                    SELECT 
                        COUNT(*) as outlier_count,
                        COUNT(CASE WHEN d > {extreme_upper} THEN 1 END) as extreme_count
                    FROM (
                        SELECT EXTRACT(EPOCH FROM (end_time_ - start_time_)) as d
                        FROM act_hi_procinst
                        WHERE proc_def_key_ = '{proc_key}'
                          AND start_time_ > NOW() - INTERVAL '{lookback_days} days'
                          AND end_time_ IS NOT NULL
                    ) sub
                    WHERE d > {upper_bound}
                """

                outlier_results = safe_execute(
                    lambda: execute_query(outlier_query),
                    default_value=[],
                    context=f"Counting outliers for {proc_key}"
                )

                outlier_count = int(outlier_results[0]['outlier_count']) if outlier_results else 0
                extreme_count = int(outlier_results[0]['extreme_count']) if outlier_results else 0
                outlier_pct = (outlier_count / total * 100) if total > 0 else 0

                # Determine severity
                if outlier_pct > high_outlier_threshold:
                    severity = 'high'
                elif outlier_pct > medium_outlier_threshold:
                    severity = 'medium'
                else:
                    severity = 'low'

                # ENHANCEMENT: Get sample outlier instances if high severity
                sample_outliers = []
                if severity in ['high', 'medium'] and outlier_count > 0:
                    sample_query = f"""
                        SELECT 
                            id_ as instance_id,
                            business_key_,
                            EXTRACT(EPOCH FROM (end_time_ - start_time_)) as duration_s,
                            start_time_,
                            end_time_
                        FROM act_hi_procinst
                        WHERE proc_def_key_ = '{proc_key}'
                          AND start_time_ > NOW() - INTERVAL '{lookback_days} days'
                          AND end_time_ IS NOT NULL
                          AND EXTRACT(EPOCH FROM (end_time_ - start_time_)) > {upper_bound}
                        ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_)) DESC
                        LIMIT {sample_limit}
                    """

                    sample_results = safe_execute(
                        lambda: execute_query(sample_query),
                        default_value=[],
                        context=f"Getting sample outliers for {proc_key}"
                    )

                    sample_outliers = [{
                        'instance_id': r['instance_id'],
                        'business_key': r.get('business_key_', 'N/A'),
                        'duration_seconds': round(float(r['duration_s']), 2),
                        'is_extreme': float(r['duration_s']) > extreme_upper,
                        'start_time': r['start_time'].isoformat() if r.get('start_time') else None,
                        'end_time': r['end_time'].isoformat() if r.get('end_time') else None
                    } for r in sample_results]

                analysis = {
                    'process_key': proc_key,
                    'total_instances': total,
                    'q1_seconds': round(q1, 2),
                    'q3_seconds': round(q3, 2),
                    'iqr_seconds': round(iqr, 2),
                    'lower_bound_seconds': round(lower_bound, 2),
                    'upper_bound_seconds': round(upper_bound, 2),
                    'extreme_threshold_seconds': round(extreme_upper, 2),  # ENHANCED
                    'avg_duration_seconds': round(avg, 2),
                    'outlier_count': outlier_count,
                    'extreme_outlier_count': extreme_count,  # ENHANCED
                    'outlier_percentage': round(outlier_pct, 2),
                    'has_extreme_outliers': extreme_count > 0,
                    'severity': severity,
                    'sample_outlier_instances': sample_outliers  # NEW
                }

                outlier_analysis.append(analysis)

                # Track high outlier processes
                if outlier_pct > high_outlier_threshold:
                    high_outlier_processes.append({
                        **analysis,
                        'message': f'{outlier_pct:.1f}% of instances are outliers - Process is HIGHLY unpredictable',
                        'recommendation': 'Do not use for SLA-critical workflows. Implement timeout safeguards and add P99 violation alerts.'
                    })

            # Sort by outlier percentage
            outlier_analysis.sort(key=lambda x: x['outlier_percentage'], reverse=True)
            high_outlier_processes.sort(key=lambda x: x['outlier_percentage'], reverse=True)

            return {
                'outlier_analysis': outlier_analysis[:max_results],
                'high_outlier_processes': high_outlier_processes,
                'total_analyzed': len(outlier_analysis),
                'high_outlier_count': len(high_outlier_processes),
                'analysis_window_days': lookback_days,
                'message': f'Analyzed {len(outlier_analysis)} processes, found {len(high_outlier_processes)} with excessive outliers (>{high_outlier_threshold}%)'
            }

        except Exception as e:
            logger.error(f"Error analyzing outlier patterns: {e}")
            return {
                'outlier_analysis': [],
                'high_outlier_processes': [],
                'total_analyzed': 0,
                'error': str(e)
            }

    def generate_critical_insights_summary(self, analysis_data: dict) -> dict:
        """
        Generate critical insights summary based on analysis (from analyze_db_data.py)
        Provides executive-ready, actionable insights
        """
        try:
            insights = {
                'extreme_variability_alerts': [],
                'version_regressions': [],
                'stuck_instance_details': [],
                'load_optimization': {},
                'high_outlier_alerts': [],
                'top_optimization_targets': []
            }

            # 1. EXTREME VARIABILITY ALERTS
            extreme_var = analysis_data.get('extreme_variability', {})
            extreme_procs = extreme_var.get('extreme_processes', [])

            for proc in extreme_procs[:3]:  # Top 3
                if proc['severity'] in ['extreme', 'high']:
                    insights['extreme_variability_alerts'].append({
                        'process_key': proc['process_key'],
                        'p95_median_ratio': proc['p95_median_ratio'],
                        'median_seconds': proc['median_seconds'],
                        'p95_seconds': proc['p95_seconds'],
                        'severity': proc['severity'],
                        'message': f"P95 is {proc['p95_median_ratio']:.0f}x median - Investigation required",
                        'recommendation': proc['recommendation'],
                        'sample_instances': proc.get('sample_slow_instances', [])
                    })

            # 2. VERSION REGRESSIONS
            version_data = analysis_data.get('version_analysis', {})
            regressions = version_data.get('regressions', [])

            for reg in regressions[:3]:  # Top 3
                if reg.get('severity') in ['critical', 'high']:
                    insights['version_regressions'].append({
                        'process_key': reg['process_key'],
                        'latest_version': reg['latest_version'],
                        'previous_version': reg['previous_version'],
                        'change_pct': reg['change_pct'],
                        'latest_avg_s': reg['latest_avg_s'],
                        'previous_avg_s': reg['previous_avg_s'],
                        'severity': reg['severity'],
                        'recommendation': f"ROLLBACK to version {reg['previous_version']} immediately"
                    })

            # 3. STUCK INSTANCE DETAILS (with business keys and sample instances)
            stuck = analysis_data.get('stuck_processes', {})
            stuck_list = stuck.get('stuck_processes', [])

            # Group by process and collect sample instances
            by_process = {}
            for s in stuck_list:
                proc = s['process_key']
                if proc not in by_process:
                    by_process[proc] = {
                        'stuck_count': 0,
                        'max_running_seconds': 0,
                        'p95_seconds': s.get('expected_p95_seconds'),
                        'severity': 'normal',
                        'sample_stuck_instances': []
                    }

                by_process[proc]['stuck_count'] += 1

                # Track max running time
                if s['stuck_for_seconds'] > by_process[proc]['max_running_seconds']:
                    by_process[proc]['max_running_seconds'] = s['stuck_for_seconds']

                # Track highest severity
                severity_order = {'critical': 4, 'warning': 3, 'attention': 2, 'normal': 1}
                if severity_order.get(s['status'], 0) > severity_order.get(by_process[proc]['severity'], 0):
                    by_process[proc]['severity'] = s['status']

                # Collect sample instances (up to 5 per process)
                if len(by_process[proc]['sample_stuck_instances']) < 5:
                    by_process[proc]['sample_stuck_instances'].append({
                        'instance_id': s['instance_id'],
                        'business_key': s['business_key'],
                        'running_seconds': s['stuck_for_seconds'],
                        'running_hours': s['stuck_for_hours'],
                        'running_days': s['stuck_for_days'],
                        'start_time': s['start_time'],
                        'status': s['status'],
                        'current_activity': s.get('current_activity')
                    })

            # Convert to list and add to insights (top 5 processes)
            for proc_key, data in sorted(by_process.items(),
                                        key=lambda x: x[1]['max_running_seconds'],
                                        reverse=True)[:5]:
                insights['stuck_instance_details'].append({
                    'process_key': proc_key,
                    'stuck_count': data['stuck_count'],
                    'max_running_seconds': data['max_running_seconds'],
                    'max_running_hours': round(data['max_running_seconds'] / 3600, 2),
                    'max_running_days': round(data['max_running_seconds'] / 86400, 2),
                    'p95_seconds': data['p95_seconds'],
                    'severity': data['severity'],
                    'sample_stuck_instances': data['sample_stuck_instances'],
                    'message': f"{data['stuck_count']} instance(s) stuck - longest running for {round(data['max_running_seconds'] / 3600, 1)}h",
                    'recommendation': self._get_stuck_recommendation(data['severity'], data['max_running_seconds'], data['stuck_count'])
                })

            # 4. LOAD OPTIMIZATION
            load_patterns = analysis_data.get('load_patterns', {})
            business_summary = load_patterns.get('business_summary', {})

            if business_summary:
                business_days = business_summary.get('business_days', {})
                weekends = business_summary.get('weekends', {})
                peak_hours = business_summary.get('peak_hours', [])

                total_instances = business_days.get('total_instances', 0) + weekends.get('total_instances', 0)
                weekend_pct = (weekends.get('total_instances', 0) / total_instances * 100) if total_instances > 0 else 0

                insights['load_optimization'] = {
                    'business_instances': business_days.get('total_instances', 0),
                    'weekend_instances': weekends.get('total_instances', 0),
                    'weekend_percentage': round(weekend_pct, 1),
                    'peak_hour': peak_hours[0]['hour'] if peak_hours else None,
                    'peak_hour_instances': peak_hours[0]['instances'] if peak_hours else 0,
                    'weekend_vs_business': business_summary.get('weekend_vs_business', {}),
                    'recommendations': business_summary.get('recommendations', [])
                }

            # 5. HIGH OUTLIER ALERTS
            outliers = analysis_data.get('outlier_patterns', {})
            high_outlier_procs = outliers.get('high_outlier_processes', [])

            for proc in high_outlier_procs[:5]:  # Top 5
                insights['high_outlier_alerts'].append({
                    'process_key': proc['process_key'],
                    'outlier_percentage': proc['outlier_percentage'],
                    'extreme_outlier_count': proc['extreme_outlier_count'],
                    'total_instances': proc['total_instances'],
                    'severity': proc['severity'],
                    'message': proc['message'],
                    'recommendation': proc['recommendation'],
                    'sample_instances': proc.get('sample_outlier_instances', [])
                })

            # 6. TOP OPTIMIZATION TARGETS
            bottlenecks = analysis_data.get('bottlenecks', {}).get('bottlenecks', [])

            for idx, bottleneck in enumerate(bottlenecks[:3], 1):  # Top 3
                if bottleneck.get('impact_hours_per_week', 0) > 10:
                    potential_savings = bottleneck['impact_hours_per_week'] * 0.1

                    insights['top_optimization_targets'].append({
                        'rank': idx,
                        'process': bottleneck['process_key'],
                        'activity': bottleneck['activity_name'],
                        'activity_type': bottleneck.get('activity_type', 'unknown'),
                        'impact_hours_per_week': bottleneck['impact_hours_per_week'],
                        'potential_savings_hours_week': round(potential_savings, 1),
                        'potential_savings_hours_year': round(potential_savings * 52, 0),
                        'avg_duration_seconds': bottleneck['avg_duration_ms'] / 1000,
                        'executions': bottleneck['executions'],
                        'recommendation': bottleneck.get('recommendation', 'Optimize this activity')
                    })

            return {
                'insights': insights,
                'total_critical_alerts': len(insights['extreme_variability_alerts']) + len(insights['version_regressions']),
                'total_stuck_processes': len(insights['stuck_instance_details']),
                'total_optimization_targets': len(insights['top_optimization_targets']),
                'overall_health': self._assess_overall_health(insights)
            }

        except Exception as e:
            logger.error(f"Error generating critical insights: {e}")
            return {
                'insights': {},
                'error': str(e)
            }

    def _assess_overall_health(self, insights: dict) -> str:
        """Assess overall system health based on insights"""
        critical_count = len(insights.get('extreme_variability_alerts', [])) + \
                         len(insights.get('version_regressions', []))

        if critical_count == 0:
            return 'excellent'
        elif critical_count <= 2:
            return 'good'
        elif critical_count <= 5:
            return 'warning'
        else:
            return 'critical'

    # =========================================================================
    # BOTTLENECK IDENTIFICATION (ENHANCED)
    # =========================================================================

    def identify_bottlenecks(self, lookback_days=None):
        """
        Identify process bottlenecks by analyzing activity durations
        Enhanced with activity types, CV, and specific recommendations
        ENHANCED: Now filters to business-critical processes only for better performance
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            max_activities = self._get_config('AI_MAX_INSTANCES', 50000)
            min_data = self._get_config('AI_MIN_DATA', 10)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)
            min_duration = self._get_config('BOTTLENECK_MIN_DURATION_SECONDS', 1.0)

            # Get business-critical process keys for filtering
            business_keys = self._get_business_critical_process_keys(lookback_days)

            process_filter = ""
            if business_keys:
                process_list = "', '".join(business_keys)
                process_filter = f"AND proc_def_key_ IN ('{process_list}')"

            query = f"""
                SELECT 
                    proc_def_key_,
                    act_id_ as activity_id,
                    act_name_ as activity_name,
                    act_type_ as activity_type,
                    EXTRACT(EPOCH FROM (end_time_ - start_time_)) * 1000 as duration_ms,
                    end_time_
                FROM act_hi_actinst
                WHERE end_time_ IS NOT NULL
                AND end_time_ > NOW() - INTERVAL '{lookback_days} days'
                AND act_id_ IS NOT NULL
                {process_filter}
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
                    'name': row['activity_name'],
                    'activity_type': row.get('activity_type', 'unknown')
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

                # Only include if average duration > configured threshold (in ms)
                if avg_duration > (min_duration * 1000):
                    # Classify activity type for better recommendations
                    activity_type = activities[0].get('activity_type', 'unknown')

                    # Calculate CV for stability
                    cv = self._compute_cv(np.std(durations), avg_duration)
                    stability = self._classify_stability(cv)

                    bottlenecks.append({
                        'process_key': proc_key,
                        'activity_id': activity_id,
                        'activity_name': activities[0]['name'] or activity_id,
                        'activity_type': activity_type,
                        'avg_duration_ms': round(float(avg_duration), 2),
                        'p95_duration_ms': round(float(p95_duration), 2),
                        'p99_duration_ms': round(float(p99_duration), 2),
                        'executions': len(activities),
                        'impact_hours_per_week': round((float(avg_duration) / 1000 / 3600) * len(activities) * (7 / lookback_days), 2),
                        'cv': round(cv, 4) if cv else None,
                        'stability': stability,
                        'recommendation': self._get_activity_recommendation(activity_type, avg_duration / 1000, len(activities))
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

    def _get_activity_recommendation(self, activity_type: str, avg_duration_s: float, executions: int) -> str:
        """Get optimization recommendation based on activity type"""
        total_hours = (avg_duration_s / 3600) * executions

        if activity_type == 'userTask':
            return f'Manual task consuming {total_hours:.1f}h - Consider automation, decision support tools, or auto-assignment rules'
        elif activity_type in ['serviceTask', 'sendTask']:
            return f'External service call taking {avg_duration_s:.1f}s avg - Consider caching, parallelization, or API optimization'
        elif activity_type == 'businessRuleTask':
            return f'Business rule execution taking {avg_duration_s:.1f}s - Review rule complexity and consider pre-computation'
        elif activity_type == 'scriptTask':
            return f'Script execution taking {avg_duration_s:.1f}s - Review script efficiency and consider optimization'
        else:
            return f'Activity consuming {total_hours:.1f}h total - Investigate for optimization opportunities'

    # =========================================================================
    # JOB FAILURE PREDICTION
    # =========================================================================

    def predict_job_failures(self, lookback_days=None):
        """
        Analyze job failure patterns and predict failure-prone jobs
        ENHANCED: Now filters to business-critical processes only for better performance
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            max_job_logs = self._get_config('AI_MAX_INSTANCES', 50000)
            min_data = self._get_config('AI_MIN_DATA', 10)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)
            critical_failure_pct = self._get_config('JOB_FAILURE_CRITICAL_PCT', 20.0)
            warning_failure_pct = self._get_config('JOB_FAILURE_WARNING_PCT', 10.0)
            monitor_failure_pct = self._get_config('JOB_FAILURE_MONITOR_PCT', 5.0)

            # Get business-critical process keys for filtering
            business_keys = self._get_business_critical_process_keys(lookback_days)

            process_filter = ""
            if business_keys:
                process_list = "', '".join(business_keys)
                process_filter = f"AND process_def_key_ IN ('{process_list}')"

            # First try standard job log approach
            query = f"""
                SELECT 
                    COALESCE(job_def_type_, 'unknown') as job_def_type_,
                    job_def_configuration_,
                    process_def_key_,
                    CASE 
                        WHEN job_exception_msg_ IS NOT NULL AND job_exception_msg_ != '' THEN 'failed'
                        ELSE 'success'
                    END as state,
                    timestamp_
                FROM act_hi_job_log
                WHERE timestamp_ > NOW() - INTERVAL '{lookback_days} days'
                  {process_filter}
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

                # For act_hi_actinst table, use proc_def_key_ not process_def_key_
                activity_process_filter = ""
                if business_keys:
                    process_list = "', '".join(business_keys)
                    activity_process_filter = f"AND proc_def_key_ IN ('{process_list}')"

                query_fallback = f"""
                    SELECT 
                        act_type_ as job_type,
                        proc_def_key_,
                        COUNT(*) as total,
                        COUNT(CASE WHEN end_time_ IS NULL THEN 1 END) as incomplete
                    FROM act_hi_actinst
                    WHERE start_time_ > NOW() - INTERVAL '{lookback_days} days'
                    AND act_type_ IN ('serviceTask', 'sendTask', 'businessRuleTask', 'scriptTask')
                    {activity_process_filter}
                    GROUP BY act_type_, proc_def_key_
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
                            'risk_level': 'high' if failure_rate > warning_failure_pct else 'medium' if failure_rate > monitor_failure_pct else 'low',
                            'recommendation': self._get_job_recommendation(failure_rate, critical_failure_pct, warning_failure_pct, monitor_failure_pct)
                        })

                    predictions.sort(key=lambda x: x['failure_rate_pct'], reverse=True)

                    total_failures = sum(p['failed_count'] for p in predictions)

                    # If all jobs are healthy (no failures), show top job types by volume
                    if total_failures == 0:
                        predictions.sort(key=lambda x: x['total_executions'], reverse=True)
                        message = f'All {len(fallback_results)} activity types healthy - no failures detected'
                    else:
                        message = f'Analyzed {len(fallback_results)} activity types from process execution (job logs unavailable)'

                    return {
                        'predictions': predictions[:max_results],
                        'total_jobs_analyzed': len(fallback_results),
                        'analysis_window_days': lookback_days,
                        'total_executions': sum(r['total'] for r in fallback_results),
                        'total_failures': total_failures,
                        'message': message,
                        'data_source': 'activity_based',
                        'all_healthy': total_failures == 0
                    }

                return {
                    'predictions': [],
                    'total_jobs_analyzed': 0,
                    'total_failures': 0,
                    'total_executions': 0,
                    'analysis_window_days': lookback_days,
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
                    'risk_level': 'high' if failure_rate > warning_failure_pct else 'medium' if failure_rate > monitor_failure_pct else 'low',
                    'recommendation': self._get_job_recommendation(failure_rate, critical_failure_pct, warning_failure_pct, monitor_failure_pct)
                })

            predictions.sort(key=lambda x: x['failure_rate_pct'], reverse=True)

            # Calculate failure type breakdown
            failure_types = defaultdict(int)
            total_failures = sum(p['failed_count'] for p in predictions)
            for pred in predictions:
                if pred['failed_count'] > 0:
                    failure_types[pred['job_type']] += pred['failed_count']

            return {
                'predictions': predictions[:max_results],
                'total_jobs_analyzed': len(job_groups),
                'analysis_window_days': lookback_days,
                'total_executions': len(results),
                'total_failures': total_failures,
                'failure_breakdown': dict(sorted(failure_types.items(), key=lambda x: x[1], reverse=True)[:10]),
                'message': f'Analyzed {len(job_groups)} job types from {len(results)} executions',
                'data_source': 'job_log'
            }

        except Exception as e:
            logger.error(f"Error predicting job failures: {e}")
            return {
                'predictions': [],
                'total_jobs_analyzed': 0,
                'total_failures': 0,
                'total_executions': 0,
                'analysis_window_days': lookback_days if 'lookback_days' in locals() else 30,
                'error': str(e),
                'message': f'Error analyzing job failures: {str(e)}',
                'data_source': 'error'
            }

    def _get_job_recommendation(self, failure_rate, critical_pct, warning_pct, monitor_pct):
        """Get recommendation based on job failure rate"""
        if failure_rate > critical_pct:
            return 'Critical: Review job configuration and increase retry attempts'
        elif failure_rate > warning_pct:
            return 'Warning: Consider adding error handling or increasing timeouts'
        elif failure_rate > monitor_pct:
            return 'Monitor: Review occasional failures for patterns'
        else:
            return 'Healthy: Job executing within normal parameters'

    # =========================================================================
    # NODE PERFORMANCE ANALYSIS
    # =========================================================================

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
            excellent_threshold = self._get_config('NODE_PERFORMANCE_EXCELLENT_THRESHOLD', 80)
            good_threshold = self._get_config('NODE_PERFORMANCE_GOOD_THRESHOLD', 60)
            poor_threshold = self._get_config('NODE_PERFORMANCE_POOR_THRESHOLD', 40)

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
                    'recommendation': self._get_node_recommendation(performance_score, excellent_threshold, good_threshold, poor_threshold)
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

    def _get_node_recommendation(self, score, excellent_threshold, good_threshold, poor_threshold):
        """Get recommendation based on node performance score"""
        if score >= excellent_threshold:
            return 'Optimal performance'
        elif score >= good_threshold:
            return 'Monitor resource usage'
        elif score >= poor_threshold:
            return 'Consider restarting or scaling'
        else:
            return 'Critical: Immediate action required'

    # =========================================================================
    # PROCESS LEADERBOARD
    # =========================================================================

    def get_process_leaderboard(self, lookback_days=None):
        """
        Performance leaderboard for process definitions
        ENHANCED: Now filters to business-critical processes only for better performance
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            min_data = self._get_config('AI_MIN_DATA', 10)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            # Get business-critical process keys for filtering
            business_keys = self._get_business_critical_process_keys(lookback_days)

            process_filter = ""
            if business_keys:
                process_list = "', '".join(business_keys)
                process_filter = f"AND proc_def_key_ IN ('{process_list}')"

            # Get grade thresholds from config
            grade_a_completion = self._get_config('PROCESS_GRADE_A_COMPLETION_PCT', 95.0)
            grade_a_failure = self._get_config('PROCESS_GRADE_A_FAILURE_PCT', 1.0)
            grade_b_completion = self._get_config('PROCESS_GRADE_B_COMPLETION_PCT', 90.0)
            grade_b_failure = self._get_config('PROCESS_GRADE_B_FAILURE_PCT', 5.0)
            grade_c_completion = self._get_config('PROCESS_GRADE_C_COMPLETION_PCT', 80.0)
            grade_c_failure = self._get_config('PROCESS_GRADE_C_FAILURE_PCT', 10.0)
            grade_d_completion = self._get_config('PROCESS_GRADE_D_COMPLETION_PCT', 70.0)

            query = f"""
                SELECT 
                    proc_def_key_,
                    COUNT(*) as instance_count,
                    AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_)) * 1000) as avg_duration_ms,
                    COUNT(*) FILTER (WHERE end_time_ IS NOT NULL) as completed_count,
                    COUNT(*) FILTER (WHERE delete_reason_ LIKE '%incident%') as failed_count
                FROM act_hi_procinst
                WHERE start_time_ > NOW() - INTERVAL '{lookback_days} days'
                  {process_filter}
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
                    'grade': self._get_process_grade(
                        completion_rate, failure_rate,
                        grade_a_completion, grade_a_failure,
                        grade_b_completion, grade_b_failure,
                        grade_c_completion, grade_c_failure,
                        grade_d_completion
                    )
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

    def _get_process_grade(self, completion_rate, failure_rate,
                          grade_a_comp, grade_a_fail,
                          grade_b_comp, grade_b_fail,
                          grade_c_comp, grade_c_fail,
                          grade_d_comp):
        """Grade process performance"""
        if completion_rate >= grade_a_comp and failure_rate < grade_a_fail:
            return 'A'
        elif completion_rate >= grade_b_comp and failure_rate < grade_b_fail:
            return 'B'
        elif completion_rate >= grade_c_comp and failure_rate < grade_c_fail:
            return 'C'
        elif completion_rate >= grade_d_comp:
            return 'D'
        else:
            return 'F'

    # =========================================================================
    # SLA BREACH PREDICTION
    # =========================================================================

    def predict_sla_breaches(self, threshold_hours=None):
        """
        Predict which active tasks are likely to breach SLA
        """
        try:
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
                'at_risk_tasks': at_risk[:max_tasks],
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

    # =========================================================================
    # ADVANCED ML ENDPOINTS
    # =========================================================================

    def find_stuck_activities_smart(self, lookback_days=None):
        """
        Advanced stuck activity detection using statistical percentile thresholds
        Identifies activities taking abnormally long based on historical patterns
        ENHANCED: Now filters to business-critical processes only for better performance
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            percentile = self._get_config('AI_STUCK_ACTIVITY_PERCENTILE', 95)
            multiplier = self._get_config('AI_STUCK_ACTIVITY_MULTIPLIER', 2.0)
            max_activities = self._get_config('AI_MAX_INSTANCES', 50000)
            min_data = self._get_config('AI_MIN_DATA', 10)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            # Get business-critical process keys for filtering
            business_keys = self._get_business_critical_process_keys(lookback_days)

            process_filter = ""
            if business_keys:
                process_list = "', '".join(business_keys)
                process_filter = f"AND proc_def_key_ IN ('{process_list}')"

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
                    {process_filter}
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

                # Determine severity using thresholds
                if duration_ratio > 5:
                    severity = 'critical'
                elif duration_ratio > 3:
                    severity = 'high'
                else:
                    severity = 'medium'

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
            stable_threshold = self._get_config('STABILITY_CV_STABLE_THRESHOLD', 0.3)
            moderate_threshold = self._get_config('STABILITY_CV_MODERATE_THRESHOLD', 1.0)
            trend_threshold = self._get_config('CAPACITY_TREND_SIGNIFICANT_THRESHOLD', 0.2)

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
            if cv < stable_threshold:
                # Stable process - median is best predictor
                predicted_duration = median_duration
                model_type = 'statistical_median'
                confidence_pct = 85
                message = f'Stable process (CV: {cv:.2f}), using median of {instance_count} instances'
            elif cv < moderate_threshold:
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
            if abs(trend_factor) > trend_threshold:  # Significant trend detected
                predicted_duration = predicted_duration * (1 + trend_factor * 0.5)  # Apply 50% of trend
                message += f' (recent trend: {trend_factor*100:+.1f}%)'

            # Calculate confidence based on sample size and variance
            sample_confidence = min(100.0, float(instance_count))
            variance_penalty = max(0.0, 30.0 - (cv * 20))
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

            trend_stable_threshold = self._get_config('CAPACITY_TREND_STABLE_THRESHOLD', 1.0)

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

                # Determine trend direction based on config threshold
                if slope > trend_stable_threshold:
                    trend = 'increasing'
                elif slope < -trend_stable_threshold:
                    trend = 'decreasing'
                else:
                    trend = 'stable'

                forecast.append({
                    'day': day,
                    'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'predicted_instances': int(predicted_daily_load),
                    'trend': trend
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
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)
            critical_failure_pct = self._get_config('JOB_FAILURE_CRITICAL_PCT', 20.0)
            warning_failure_pct = self._get_config('JOB_FAILURE_WARNING_PCT', 10.0)

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
                    if failure_rate > critical_failure_pct or abs(duration_impact) > 50:
                        impact_level = 'high'
                    else:
                        impact_level = 'medium'

                    variable_impacts.append({
                        'variable_name': var_name,
                        'variable_type': var_data['variable_type'].iloc[0],
                        'total_instances': int(total_instances),
                        'failure_rate_pct': round(failure_rate, 1),
                        'duration_impact_pct': round(duration_impact, 1),
                        'impact_level': impact_level,
                        'recommendation': self._get_variable_recommendation(var_name, failure_rate, duration_impact, critical_failure_pct, warning_failure_pct),
                        'sample_values': list(var_data['variable_value'].unique()[:5])
                    })

            # Sort by impact
            variable_impacts.sort(key=lambda x: abs(x['failure_rate_pct']) + abs(x['duration_impact_pct']), reverse=True)

            return {
                'variable_impacts': variable_impacts[:max_results],
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

    def _get_variable_recommendation(self, var_name: str, failure_rate: float, duration_impact: float, critical_pct: float, warning_pct: float) -> str:
        """Generate recommendation based on variable impact"""
        if failure_rate > critical_pct:
            return f'Critical: Variable {var_name} strongly correlates with failures - investigate business logic'
        elif failure_rate > warning_pct:
            return f'Warning: Variable {var_name} correlates with failures - review validation rules'
        elif abs(duration_impact) > 50:
            return f'Performance: Variable {var_name} significantly impacts duration - consider optimization'
        elif abs(duration_impact) > 20:
            return f'Monitor: Variable {var_name} affects duration - track for patterns'
        else:
            return f'Informational: Variable {var_name} shows measurable impact'

    # =========================================================================
    # PROFESSIONAL INSIGHTS GENERATOR
    # =========================================================================

    def generate_professional_insights(self, analysis_data: dict) -> dict:
        """
        Generate professional enterprise insights from comprehensive analysis
        Provides executive-ready recommendations with impact calculations
        """
        try:
            insights = {
                'critical_alerts': [],
                'optimization_opportunities': [],
                'capacity_recommendations': [],
                'deployment_quality': {},
                'summary_stats': {}
            }

            extreme_var_high_threshold = self._get_config('EXTREME_VARIABILITY_RATIO_HIGH', 50.0)
            max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)

            # 1. CRITICAL ALERTS - Extreme Variability
            extreme_var = analysis_data.get('extreme_variability', {})
            extreme_procs = extreme_var.get('extreme_processes', [])

            if extreme_procs:
                for proc in extreme_procs[:3]:  # Top 3
                    if proc['severity'] in ['extreme', 'high']:
                        insights['critical_alerts'].append({
                            'priority': 'critical',
                            'category': 'performance_risk',
                            'process': proc['process_key'],
                            'issue': f"P95 is {proc['p95_median_ratio']:.0f}x median duration",
                            'impact': 'UNSUITABLE FOR SLA-CRITICAL WORKFLOWS',
                            'action': proc['recommendation'],
                            'severity': proc['severity']
                        })

            # 2. CRITICAL ALERTS - Version Regressions
            version_data = analysis_data.get('version_analysis', {})
            regressions = version_data.get('regressions', [])

            for regression in regressions[:3]:  # Top 3 worst
                if regression.get('severity') in ['critical', 'high']:
                    insights['critical_alerts'].append({
                        'priority': 'critical',
                        'category': 'deployment_regression',
                        'process': regression['process_key'],
                        'issue': f"Version {regression['latest_version']} is {regression['change_pct']:.1f}% slower",
                        'impact': f"Degradation: v{regression['previous_version']} ({regression['previous_avg_s']:.1f}s) → v{regression['latest_version']} ({regression['latest_avg_s']:.1f}s)",
                        'action': f"ROLLBACK to version {regression['previous_version']} immediately",
                        'severity': regression['severity']
                    })

            # 3. OPTIMIZATION OPPORTUNITIES - Bottlenecks
            bottlenecks = analysis_data.get('bottlenecks', {}).get('bottlenecks', [])

            for idx, bottleneck in enumerate(bottlenecks[:5], 1):  # Top 5
                if bottleneck.get('impact_hours_per_week', 0) > 10:
                    potential_savings = bottleneck['impact_hours_per_week'] * 0.1  # 10% improvement

                    insights['optimization_opportunities'].append({
                        'rank': idx,
                        'process': bottleneck['process_key'],
                        'activity': bottleneck['activity_name'],
                        'activity_type': bottleneck.get('activity_type', 'unknown'),
                        'current_impact_hours_per_week': bottleneck['impact_hours_per_week'],
                        'potential_savings_hours_per_week': round(potential_savings, 1),
                        'potential_savings_hours_per_year': round(potential_savings * 52, 0),
                        'avg_duration_seconds': bottleneck['avg_duration_ms'] / 1000,
                        'executions': bottleneck['executions'],
                        'stability': bottleneck.get('stability', 'unknown'),
                        'recommendation': bottleneck.get('recommendation', 'Investigate and optimize')
                    })

            # 4. CAPACITY RECOMMENDATIONS - Load Patterns
            load_patterns = analysis_data.get('load_patterns', {})
            business_summary = load_patterns.get('business_summary', {})

            if business_summary:
                recommendations = business_summary.get('recommendations', [])
                insights['capacity_recommendations'] = recommendations

                # Add custom recommendations based on data
                business_days = business_summary.get('business_days', {})
                weekends = business_summary.get('weekends', {})
                peak_hours = business_summary.get('peak_hours', [])

                total_instances = business_days.get('total_instances', 0) + weekends.get('total_instances', 0)
                weekend_pct = (weekends.get('total_instances', 0) / total_instances * 100) if total_instances > 0 else 0

                insights['capacity_recommendations'].append({
                    'priority': 'medium',
                    'category': 'scheduling',
                    'message': f'Weekend utilization is {weekend_pct:.1f}% (Total: {total_instances:,} instances)',
                    'details': {
                        'business_days_instances': business_days.get('total_instances', 0),
                        'weekend_instances': weekends.get('total_instances', 0),
                        'peak_hour': peak_hours[0]['hour'] if peak_hours else None
                    }
                })

            # 5. DEPLOYMENT QUALITY ASSESSMENT
            categories_data = analysis_data.get('categories', {})
            category_counts = categories_data.get('category_counts', {})
            total_processes = categories_data.get('total_processes', 0)

            improvements = version_data.get('improvements', [])

            insights['deployment_quality'] = {
                'total_processes': total_processes,
                'by_category': category_counts,
                'version_changes': {
                    'regressions': len(regressions),
                    'improvements': len(improvements),
                    'regression_rate_pct': round((len(regressions) / max(len(regressions) + len(improvements), 1)) * 100, 1)
                },
                'quality_score': self._calculate_deployment_quality_score(regressions, improvements, extreme_procs),
                'grade': self._get_deployment_grade(len(regressions), len(improvements))
            }

            # 6. SUMMARY STATISTICS
            anomalies = analysis_data.get('anomalies', {}).get('anomalies', [])
            incidents = analysis_data.get('incidents', {}).get('patterns', [])
            stuck = analysis_data.get('stuck_activities', {}).get('stuck_activities', [])
            stuck_processes = analysis_data.get('stuck_processes', {}).get('stuck_processes', [])

            insights['summary_stats'] = {
                'processes_analyzed': total_processes,
                'anomalies_detected': len(anomalies),
                'incident_patterns': len(incidents),
                'stuck_activities': len(stuck),
                'stuck_processes': len(stuck_processes),
                'regressions': len(regressions),
                'extreme_variability_processes': len(extreme_procs),
                'optimization_targets': len(bottlenecks),
                'total_critical_alerts': len(insights['critical_alerts']),
                'health_status': self._get_overall_health_status(insights)
            }

            return insights

        except Exception as e:
            logger.error(f"Error generating professional insights: {e}")
            return {
                'critical_alerts': [],
                'optimization_opportunities': [],
                'capacity_recommendations': [],
                'deployment_quality': {},
                'summary_stats': {},
                'error': str(e)
            }

    def _calculate_deployment_quality_score(self, regressions, improvements, extreme_procs) -> float:
        """Calculate deployment quality score (0-100)"""
        # Start at 100
        score = 100.0

        # Penalize regressions
        for regression in regressions:
            severity = regression.get('severity', 'low')
            if severity == 'critical':
                score -= 10
            elif severity == 'high':
                score -= 5
            elif severity == 'medium':
                score -= 2

        # Reward improvements
        score += len(improvements) * 1

        # Penalize extreme variability
        for proc in extreme_procs:
            if proc.get('severity') == 'extreme':
                score -= 3
            elif proc.get('severity') == 'high':
                score -= 1

        return max(0, min(100, score))

    def _get_deployment_grade(self, regressions_count: int, improvements_count: int) -> str:
        """Get deployment quality grade"""
        if regressions_count == 0 and improvements_count > 0:
            return 'A - Excellent'
        elif regressions_count == 0:
            return 'B - Good'
        elif regressions_count <= 2:
            return 'C - Fair'
        elif regressions_count <= 5:
            return 'D - Poor'
        else:
            return 'F - Critical'

    def _get_overall_health_status(self, insights: dict) -> str:
        """Determine overall system health status"""
        critical_count = len(insights.get('critical_alerts', []))

        if critical_count == 0:
            return 'excellent'
        elif critical_count <= 2:
            return 'good'
        elif critical_count <= 5:
            return 'warning'
        else:
            return 'critical'

    # =========================================================================
    # AI RECOMMENDATIONS
    # =========================================================================

    def get_ai_recommendations(self, analysis_results):
        """
        Generate actionable AI recommendations based on all analysis
        """
        recommendations = []
        max_recommendations = self._get_config('AI_UI_RESULTS_LIMIT', 20)

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

        return recommendations[:max_recommendations]

    # =========================================================================
    # CRITICAL INSIGHTS GENERATION
    # =========================================================================

    def generate_critical_insights(self, analysis_data=None):
        """
        Generate critical insights with specific instance IDs and business keys
        Based on analyze_db_data.py generate_critical_insights() function

        Provides actionable recommendations with concrete examples:
        - Extreme variability warnings (P95/median > 100x)
        - Version regression alerts (>100% slower)
        - Stuck instance deep dive with oldest per process
        - Load optimization opportunities
        - Outlier-heavy process warnings (>15%)
        - Top 3 activity optimization targets with ROI
        """
        try:
            insights = {
                'extreme_variability': [],
                'version_regressions': [],
                'stuck_instances': [],
                'load_optimizations': [],
                'outlier_warnings': [],
                'activity_targets': [],
                'timestamp': datetime.now().isoformat()
            }

            # If no analysis data provided, gather it
            if not analysis_data:
                analysis_data = {}

            # 1. EXTREME VARIABILITY DETECTION (P95 >> Median)
            if 'extreme_variability' in analysis_data:
                for proc in analysis_data['extreme_variability'].get('extreme_processes', [])[:5]:
                    if proc.get('variance_ratio', 0) > 100:
                        insights['extreme_variability'].append({
                            'process_key': proc['process_key'],
                            'median_s': proc['median_s'],
                            'p95_s': proc['p95_s'],
                            'ratio': proc['variance_ratio'],
                            'severity': proc['severity'],
                            'sample_instance_id': proc.get('sample_instance_id'),
                            'sample_business_key': proc.get('sample_business_key'),
                            'recommendation': f"Investigate why some instances take {proc['variance_ratio']:.0f}x longer. Check for external dependencies, stuck subprocesses, or consider splitting into fast/slow paths.",
                            'actions': [
                                'Check for external dependencies causing delays',
                                'Look for error retries or stuck subprocess',
                                'Consider splitting into fast-path and slow-path variants'
                            ]
                        })

            # 2. VERSION REGRESSIONS (>100% slower)
            if 'version_analysis' in analysis_data:
                for reg in analysis_data['version_analysis'].get('regressions', [])[:3]:
                    if reg.get('change_pct', 0) > 100:
                        insights['version_regressions'].append({
                            'process_key': reg['process_key'],
                            'latest_version': reg['latest_version'],
                            'previous_version': reg['previous_version'],
                            'degradation_pct': reg['change_pct'],
                            'latest_avg_s': reg['latest_avg_s'],
                            'previous_avg_s': reg['previous_avg_s'],
                            'sample_instance_id': reg.get('sample_instance_id'),
                            'sample_business_key': reg.get('sample_business_key'),
                            'severity': 'critical',
                            'recommendation': f"ROLLBACK to version {reg['previous_version']} immediately - Version {reg['latest_version']} is {reg['change_pct']:.1f}% slower",
                            'actions': [
                                'Investigate code changes between versions',
                                'Check database query performance',
                                'Review external API changes',
                                f"Consider rollback to v{reg['previous_version']}"
                            ]
                        })

            # 3. STUCK INSTANCE DEEP DIVE
            if 'stuck_processes' in analysis_data:
                stuck_by_process = {}
                for stuck in analysis_data['stuck_processes'].get('stuck_processes', []):
                    proc_key = stuck['process_key']
                    if proc_key not in stuck_by_process:
                        stuck_by_process[proc_key] = []
                    stuck_by_process[proc_key].append(stuck)

                for proc_key, stuck_list in list(stuck_by_process.items())[:3]:
                    oldest = max(stuck_list, key=lambda x: x['stuck_for_hours'])
                    insights['stuck_instances'].append({
                        'process_key': proc_key,
                        'stuck_count': len(stuck_list),
                        'oldest_hours': oldest['stuck_for_hours'],
                        'oldest_days': oldest['stuck_for_days'],
                        'oldest_instance_id': oldest['instance_id'],
                        'oldest_business_key': oldest['business_key'],
                        'severity': 'critical' if len(stuck_list) > 50 else 'warning',
                        'recommendation': f"{len(stuck_list)} instance(s) stuck for up to {oldest['stuck_for_days']:.0f} days",
                        'actions': [
                            'Check for waiting external callbacks',
                            'Look for missing message correlation',
                            'Review stuck user tasks or timers',
                            f"Investigate instance {oldest['instance_id'][:16]}... | BKey: {oldest['business_key']}"
                        ]
                    })

            # 4. LOAD PATTERN OPTIMIZATIONS
            if 'load_patterns' in analysis_data:
                load_data = analysis_data['load_patterns'].get('business_summary', {})
                weekend_data = load_data.get('weekends', {})
                business_data = load_data.get('business_days', {})

                if weekend_data.get('total_instances', 0) > 0 and business_data.get('total_instances', 0) > 0:
                    total = weekend_data['total_instances'] + business_data['total_instances']
                    weekend_ratio = (weekend_data['total_instances'] / total) * 100

                    if weekend_ratio < 20:
                        insights['load_optimizations'].append({
                            'type': 'weekend_underutilization',
                            'weekend_pct': round(weekend_ratio, 1),
                            'severity': 'info',
                            'recommendation': f"Weekend load is only {weekend_ratio:.1f}% of total traffic - schedule batch jobs on weekends",
                            'actions': [
                                'Schedule data cleanup processes on weekends',
                                'Move report generation to weekend hours',
                                'Run archive operations during low-load periods'
                            ]
                        })

                peak_hours = load_data.get('peak_hours', [])
                if peak_hours:
                    peak = peak_hours[0]
                    insights['load_optimizations'].append({
                        'type': 'peak_hour_optimization',
                        'peak_hour': peak['hour'],
                        'peak_instances': peak['instances'],
                        'severity': 'info',
                        'recommendation': f"Peak load at {peak['hour']:02d}:00 with {peak['instances']:,} instances",
                        'actions': [
                            f"Pre-warm caches before {peak['hour']:02d}:00",
                            f"Scale up resources at {peak['hour']-1:02d}:00",
                            f"Delay non-critical jobs until after {peak['hour']+2:02d}:00"
                        ]
                    })

            # 5. OUTLIER-HEAVY PROCESSES (>15%)
            if 'outlier_patterns' in analysis_data:
                for outlier in analysis_data['outlier_patterns'].get('outlier_analysis', [])[:5]:
                    if outlier.get('outlier_pct', 0) > 15:
                        insights['outlier_warnings'].append({
                            'process_key': outlier['process_key'],
                            'outlier_pct': outlier['outlier_pct'],
                            'outlier_count': outlier['outlier_count'],
                            'total_count': outlier['total_count'],
                            'threshold_s': outlier.get('iqr_upper_threshold_s', 0),
                            'sample_outliers': outlier.get('sample_outliers', [])[:3],
                            'severity': 'high' if outlier['outlier_pct'] > 20 else 'medium',
                            'recommendation': f"Process is HIGHLY unpredictable with {outlier['outlier_pct']:.1f}% outliers",
                            'actions': [
                                "Don't use for SLA-critical workflows",
                                'Implement timeout safeguards',
                                'Add monitoring alerts for P99 violations',
                                'Investigate root cause of variability'
                            ]
                        })

            # 6. TOP 3 ACTIVITY OPTIMIZATION TARGETS (with ROI)
            if 'activity_bottlenecks' in analysis_data:
                activities = analysis_data['activity_bottlenecks'].get('activities', [])
                # Sort by potential savings
                sorted_activities = sorted(activities, key=lambda x: x.get('potential_savings_hours', 0), reverse=True)[:3]

                for idx, act in enumerate(sorted_activities, 1):
                    insights['activity_targets'].append({
                        'rank': idx,
                        'process_key': act['process_key'],
                        'activity_name': act['activity_name'],
                        'activity_type': act['activity_type'],
                        'execution_count': act['execution_count'],
                        'avg_duration_s': act['avg_duration_s'],
                        'p95_duration_s': act['p95_duration_s'],
                        'total_time_hours': act['total_time_hours'],
                        'potential_savings_hours': act['potential_savings_hours'],
                        'sample_instance_id': act.get('sample_instance_id'),
                        'sample_business_key': act.get('sample_business_key'),
                        'severity': 'info',
                        'recommendation': f"10% improvement saves {act['potential_savings_hours']:.0f} hours",
                        'actions': [
                            'Reduce manual work with automation' if act['activity_type'] == 'userTask' else 'Optimize service call performance',
                            'Add decision support tools' if act['activity_type'] == 'userTask' else 'Consider caching or parallelization',
                            f"Investigate sample: {act.get('sample_instance_id', 'N/A')[:16]}... | BKey: {act.get('sample_business_key', 'N/A')}"
                        ]
                    })

            # Calculate summary statistics
            insights['summary'] = {
                'critical_issues': len(insights['version_regressions']) + len([s for s in insights['stuck_instances'] if s['severity'] == 'critical']),
                'warnings': len(insights['extreme_variability']) + len(insights['outlier_warnings']),
                'optimization_opportunities': len(insights['activity_targets']) + len(insights['load_optimizations']),
                'total_potential_savings_hours': sum(a['potential_savings_hours'] for a in insights['activity_targets'])
            }

            logger.info(f"Generated critical insights: {insights['summary']['critical_issues']} critical, {insights['summary']['warnings']} warnings")
            return insights

        except Exception as e:
            logger.error(f"Error generating critical insights: {e}")
            return {
                'extreme_variability': [],
                'version_regressions': [],
                'stuck_instances': [],
                'load_optimizations': [],
                'outlier_warnings': [],
                'activity_targets': [],
                'summary': {'critical_issues': 0, 'warnings': 0, 'optimization_opportunities': 0},
                'error': str(e)
            }

    # =========================================================================
    # ALIAS FUNCTIONS (for API compatibility)
    # =========================================================================

    def detect_extreme_variability(self, lookback_days=None):
        """
        Alias for analyze_extreme_variability (for API endpoint compatibility)
        """
        # Get process categories first
        cat_result = self.get_process_categories(lookback_days=lookback_days)
        process_categories = cat_result.get('categories', {})

        # Call the main analysis function
        return self.analyze_extreme_variability(process_categories=process_categories)


# Singleton instance
_ai_analytics = AIAnalytics()


def get_ai_analytics():
    """Get the AI analytics singleton"""
    return _ai_analytics
