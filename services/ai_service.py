"""
AI/ML Service
Intelligent analytics and predictions for Camunda cluster health
Uses mainly historical data from ACT_HI_* and ACT_RU_* tables
"""
import logging
import numpy as np
from collections import defaultdict
from typing import Optional
from helpers.error_handler import safe_execute
from helpers.db_helper import execute_query
from services.database_service import timed_cache
from datetime import datetime, timedelta

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

    @timed_cache(seconds=3600)
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

    @timed_cache(seconds=300)
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

    @timed_cache(seconds=3600)
    def get_process_categories(self, lookback_days=None):
        """
        Categorize all processes by duration
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
    # ANOMALY DETECTION
    # =========================================================================

    @timed_cache(seconds=3600)
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
    # INCIDENT PATTERN ANALYSIS
    # =========================================================================

    @timed_cache(seconds=1800)
    def get_comprehensive_incident_health(self, lookback_days=None):
        """
        IMPROVED: Intelligently adapts to Camunda configuration
        Automatically detects if history is enabled and chooses best data source
        """
        try:
            if lookback_days is None:
                lookback_days = self._get_config('AI_LOOKBACK_DAYS', 30)

            # Step 1: Always get current runtime incidents
            current_incidents = self._get_runtime_incidents(lookback_days)

            # Step 2: Detect if history is enabled and get historical data
            historical_data = self._get_historical_incidents_smart(lookback_days)

            # Step 3: Merge runtime and historical (avoiding duplicates)
            all_incidents = self._merge_incidents(current_incidents, historical_data)

            # Step 4: Identify recurring issues
            recurring_issues = self._identify_recurring_incidents_smart(
                current_incidents,
                historical_data
            )

            # Step 5: Score all incidents
            scored_incidents = self._score_incident_severity(current_incidents)

            # Step 6: Generate recommendations
            recommendations = self._generate_incident_recommendations(
                scored_incidents,
                recurring_issues,
                historical_data
            )

            return {
                'current_state': {
                    'active_incidents': scored_incidents,
                    'total_active': len(current_incidents),
                    'by_severity': self._group_by_severity(scored_incidents),
                    'by_process': self._group_by_process(scored_incidents),
                    'oldest_incident_hours': self._get_oldest_incident_age(current_incidents)
                },
                'historical_trends': {
                    'patterns': historical_data.get('patterns', [])[:10],
                    'total_incidents': historical_data.get('total_incidents', 0),
                    'resolution_rate_pct': self._calculate_resolution_rate(historical_data),
                    'avg_resolution_time_hours': self._calculate_avg_resolution_time(historical_data),
                    'data_source': historical_data.get('data_source', 'unknown'),
                    'history_enabled': historical_data.get('history_enabled', False)
                },
                'recurring_issues': recurring_issues,
                'recommendations': recommendations,
                'health_status': self._assess_incident_health(scored_incidents, historical_data),
                'analysis_window_days': lookback_days,
                'configuration': {
                    'history_enabled': historical_data.get('history_enabled', False),
                    'data_sources_used': historical_data.get('data_sources_used', []),
                    'total_data_points': len(all_incidents)
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting comprehensive incident health: {e}", exc_info=True)
            return {
                'current_state': {'active_incidents': [], 'total_active': 0},
                'historical_trends': {},
                'recurring_issues': [],
                'recommendations': [],
                'error': str(e)
            }

    def _get_historical_incidents_smart(self, lookback_days):
        """
        Automatically detect which data sources are available
        Returns unified structure regardless of configuration
        """
        max_incidents = self._get_config('AI_MAX_INCIDENTS', 1000)
        max_results = self._get_config('AI_UI_RESULTS_LIMIT', 20)
        sample_limit = self._get_config('INCIDENT_PATTERN_SAMPLE_LIMIT', 5)

        data_sources_used = []

        # ===================================================================
        # SOURCE 1: Try historical incidents (if history is enabled)
        # ===================================================================
        historical_incidents = self._try_get_historical_incidents(lookback_days, max_incidents)

        if historical_incidents and len(historical_incidents) > 0:
            logger.info(f"Found {len(historical_incidents)} historical incidents")
            data_sources_used.append('historical_incidents')

            # Process historical incidents
            patterns = self._process_incidents_to_patterns(
                historical_incidents,
                sample_limit,
                data_source='historical'
            )

            return {
                'patterns': patterns[:max_results],
                'total_incidents': len(historical_incidents),
                'unique_patterns': len(patterns),
                'total_open': sum(p['open_count'] for p in patterns),
                'total_resolved': sum(p['resolved_count'] for p in patterns),
                'root_cause_categories': self._extract_root_causes(patterns),
                'analysis_window_days': lookback_days,
                'data_source': 'historical',
                'history_enabled': True,
                'data_sources_used': data_sources_used,
                'message': f'Found {len(patterns)} patterns from {len(historical_incidents)} historical incidents'
            }

        # ===================================================================
        # SOURCE 2: Fallback to job exceptions + failed activities
        # This works regardless of history configuration
        # ===================================================================
        logger.info("No historical incidents - using job logs and activity data")

        # Get job exceptions (always available)
        job_exceptions = self._try_get_job_exceptions(lookback_days, max_incidents)
        if job_exceptions:
            data_sources_used.append('job_exceptions')
            logger.info(f"Found {len(job_exceptions)} job exceptions")

        # Get failed activities (always available)
        failed_activities = self._try_get_failed_activities(lookback_days, max_incidents)
        if failed_activities:
            data_sources_used.append('failed_activities')
            logger.info(f"Found {len(failed_activities)} failed activities")

        # Merge job exceptions and failed activities
        combined_incidents = (job_exceptions or []) + (failed_activities or [])

        if not combined_incidents:
            return {
                'patterns': [],
                'total_incidents': 0,
                'unique_patterns': 0,
                'analysis_window_days': lookback_days,
                'data_source': 'none',
                'history_enabled': False,
                'data_sources_used': [],
                'message': f'✓ No incidents found in last {lookback_days} days - System is healthy!',
                'health_status': 'excellent'
            }

        # Process combined incidents
        patterns = self._process_incidents_to_patterns(
            combined_incidents,
            sample_limit,
            data_source='runtime_combined'
        )

        return {
            'patterns': patterns[:max_results],
            'total_incidents': len(combined_incidents),
            'unique_patterns': len(patterns),
            'total_open': sum(p['open_count'] for p in patterns),
            'total_resolved': sum(p['resolved_count'] for p in patterns),
            'root_cause_categories': self._extract_root_causes(patterns),
            'analysis_window_days': lookback_days,
            'data_source': 'runtime_combined',
            'history_enabled': False,
            'data_sources_used': data_sources_used,
            'message': f'Found {len(patterns)} patterns from {len(combined_incidents)} runtime data points (history disabled)'
        }

    def _try_get_historical_incidents(self, lookback_days, max_incidents):
        """Try to get historical incidents - returns None if not available"""
        query = f"""
            SELECT 
                inc.id_ as incident_id,
                inc.incident_type_,
                inc.incident_msg_,
                inc.proc_def_key_,
                inc.activity_id_,
                inc.create_time_,
                inc.end_time_,
                inc.proc_inst_id_,
                pi.business_key_,
                CASE 
                    WHEN inc.end_time_ IS NULL THEN 'open' 
                    WHEN inc.incident_state_ = 2 THEN 'resolved'
                    WHEN inc.removal_time_ IS NOT NULL THEN 'deleted'
                    ELSE 'resolved' 
                END as status,
                EXTRACT(EPOCH FROM (COALESCE(inc.end_time_, NOW()) - inc.create_time_)) as duration_seconds
            FROM act_hi_incident inc
            LEFT JOIN act_hi_procinst pi ON inc.proc_inst_id_ = pi.id_
            WHERE inc.create_time_ > NOW() - INTERVAL '{lookback_days} days'
            ORDER BY inc.create_time_ DESC
            LIMIT {max_incidents}
        """

        try:
            results = execute_query(query)
            return results if results and len(results) > 0 else None
        except Exception as e:
            logger.debug(f"Historical incidents not available: {e}")
            return None

    def _try_get_job_exceptions(self, lookback_days, max_incidents):
        """Get job exceptions"""
        query = f"""
            SELECT 
                jl.id_ as incident_id,
                COALESCE(jl.job_def_type_, 'failedJob') as incident_type_,
                jl.job_exception_msg_ as incident_msg_,
                jl.process_def_key_ as proc_def_key_,
                jl.act_id_ as activity_id_,
                jl.timestamp_ as create_time_,
                NULL as end_time_,
                jl.process_instance_id_ as proc_inst_id_,
                pi.business_key_,
                CASE 
                    WHEN jl.job_state_ = 2 THEN 'resolved'
                    WHEN jl.job_retries_ = 0 THEN 'open'
                    ELSE 'retrying'
                END as status,
                0 as duration_seconds
            FROM act_hi_job_log jl
            LEFT JOIN act_ru_execution pi ON jl.process_instance_id_ = pi.proc_inst_id_
            WHERE jl.timestamp_ > NOW() - INTERVAL '{lookback_days} days'
              AND jl.job_exception_msg_ IS NOT NULL
              AND jl.job_exception_msg_ != ''
            ORDER BY jl.timestamp_ DESC
            LIMIT {max_incidents}
        """

        try:
            results = execute_query(query)
            return results if results and len(results) > 0 else None
        except Exception as e:
            logger.warning(f"Job exceptions query failed: {e}")
            return None

    def _try_get_failed_activities(self, lookback_days, max_incidents):
        """Get failed/stuck activities"""
        query = f"""
            SELECT 
                ai.id_ as incident_id,
                ai.act_type_ as incident_type_,
                CONCAT('Activity incomplete: ', ai.act_name_) as incident_msg_,
                ai.proc_def_key_,
                ai.act_id_ as activity_id_,
                ai.start_time_ as create_time_,
                ai.end_time_,
                ai.proc_inst_id_,
                pi.business_key_,
                CASE 
                    WHEN ai.end_time_ IS NULL THEN 'open' 
                    ELSE 'resolved' 
                END as status,
                EXTRACT(EPOCH FROM (COALESCE(ai.end_time_, NOW()) - ai.start_time_)) as duration_seconds
            FROM act_hi_actinst ai
            LEFT JOIN act_ru_execution pi ON ai.proc_inst_id_ = pi.proc_inst_id_
            WHERE ai.start_time_ > NOW() - INTERVAL '{lookback_days} days'
              AND ai.end_time_ IS NULL
              AND ai.act_type_ IN ('serviceTask', 'sendTask', 'businessRuleTask', 'scriptTask')
            ORDER BY ai.start_time_ DESC
            LIMIT {max_incidents}
        """

        try:
            results = execute_query(query)
            return results if results and len(results) > 0 else None
        except Exception as e:
            logger.warning(f"Failed activities query failed: {e}")
            return None

    def _process_incidents_to_patterns(self, incidents, sample_limit, data_source):
        """
        Unified processing: Convert raw incidents to patterns
        Works with any incident source (historical, jobs, activities)
        """
        pattern_groups = defaultdict(lambda: {
            'incidents': [],
            'open_count': 0,
            'resolved_count': 0,
            'retrying_count': 0,
            'total_duration': 0,
            'sample_instances': []
        })

        for row in incidents:
            incident_type = row.get('incident_type_') or 'Unknown Type'
            incident_msg = (row.get('incident_msg_') or 'No error message')[:100]
            key = f"{incident_type}:::{incident_msg}"

            pattern_groups[key]['incidents'].append(row)

            # Track status (handle all possible states)
            status = row.get('status', 'unknown')
            if status == 'open':
                pattern_groups[key]['open_count'] += 1
            elif status == 'resolved':
                pattern_groups[key]['resolved_count'] += 1
            elif status == 'retrying':
                pattern_groups[key]['retrying_count'] += 1

            # Track duration
            duration = self._safe_float(row.get('duration_seconds'))
            if duration and duration > 0:
                pattern_groups[key]['total_duration'] += duration

            # Store sample instances
            if len(pattern_groups[key]['sample_instances']) < sample_limit:
                pattern_groups[key]['sample_instances'].append({
                    'instance_id': row.get('proc_inst_id_', 'N/A'),
                    'business_key': row.get('business_key_', 'N/A'),
                    'status': status,
                    'create_time': row['create_time'].isoformat() if row.get('create_time') else None
                })

        # Convert to pattern list
        patterns = []
        for key, data in pattern_groups.items():
            parts = key.split(':::', 1)
            incident_type = parts[0] if len(parts) > 0 else 'Unknown'
            msg = parts[1] if len(parts) > 1 else 'No message'

            incidents = data['incidents']
            affected_processes = list(set([i.get('proc_def_key_') for i in incidents if i.get('proc_def_key_')]))
            affected_activities = list(set([i.get('activity_id_') for i in incidents if i.get('activity_id_')]))

            timestamps = [i['create_time'] for i in incidents if i.get('create_time')]
            first_seen = min(timestamps).isoformat() if timestamps else None
            last_seen = max(timestamps).isoformat() if timestamps else None

            total_count = len(incidents)
            avg_duration = (data['total_duration'] / total_count) if total_count > 0 else 0

            # Root cause analysis
            root_cause = self._categorize_incident_root_cause(incident_type, msg)

            patterns.append({
                'incident_type': incident_type,
                'error_message': msg,
                'occurrence_count': total_count,
                'open_count': data['open_count'],
                'resolved_count': data['resolved_count'],
                'retrying_count': data.get('retrying_count', 0),
                'avg_duration_hours': round(avg_duration / 3600, 2) if avg_duration > 0 else 0,
                'affected_processes': affected_processes[:5],
                'affected_activities': affected_activities[:5],
                'first_seen': first_seen,
                'last_seen': last_seen,
                'frequency_per_day': round(total_count / 30, 2),  # Rough estimate
                'sample_instances': data['sample_instances'],
                'root_cause': root_cause,
                'source': data_source,
                'health_status': self._classify_pattern_health(
                    data['open_count'],
                    data['resolved_count'],
                    total_count
                )
            })

        # Sort by occurrence count
        patterns.sort(key=lambda x: x['occurrence_count'], reverse=True)
        return patterns

    def _classify_pattern_health(self, open_count, resolved_count, total_count):
        """Classify health status of a pattern"""
        if total_count == 0:
            return 'unknown'

        open_pct = (open_count / total_count) * 100

        if open_pct > 50:
            return 'critical'
        elif open_pct > 20:
            return 'degraded'
        elif open_count > 0:
            return 'warning'
        else:
            return 'healthy'

    def _merge_incidents(self, current_incidents, historical_data):
        """
        Merge runtime and historical incidents, removing duplicates
        """
        all_incidents = list(current_incidents)  # Start with runtime

        # Add historical patterns (if available)
        historical_patterns = historical_data.get('patterns', [])
        for pattern in historical_patterns:
            for sample in pattern.get('sample_instances', []):
                # Check if not already in current incidents
                instance_id = sample.get('instance_id')
                if instance_id and instance_id != 'N/A':
                    if not any(inc.get('proc_inst_id_') == instance_id for inc in all_incidents):
                        # Add as synthetic incident
                        all_incidents.append({
                            'proc_inst_id_': instance_id,
                            'incident_type_': pattern['incident_type'],
                            'incident_msg_': pattern['error_message'],
                            'business_key': sample.get('business_key'),
                            'status': sample.get('status'),
                            'source': 'historical'
                        })

        return all_incidents

    def _identify_recurring_incidents_smart(self, current_incidents, historical_data):
        """
        Works with both historical and runtime data
        Identifies true recurring issues regardless of configuration
        """
        if not current_incidents:
            return []

        patterns = historical_data.get('patterns', [])
        if not patterns:
            # No historical data - can't identify recurring
            # But can identify repeated runtime incidents
            return self._identify_repeated_runtime_incidents(current_incidents)

        # Original logic for when have historical data
        recurring = []
        similarity_threshold = self._get_config('INCIDENT_SIMILARITY_THRESHOLD', 0.7)

        for current in current_incidents:
            current_msg = (current.get('incident_msg_') or '')[:100]
            current_type = current.get('incident_type_', 'unknown')

            for pattern in patterns:
                pattern_type = pattern.get('incident_type', '')
                pattern_msg = (pattern.get('error_message') or '')[:100]

                if pattern_type == current_type:
                    similarity = self._calculate_similarity(current_msg, pattern_msg)

                    if similarity > similarity_threshold:
                        recurring.append({
                            'incident_id': current.get('incident_id'),
                            'incident_type': current_type,
                            'error_message': current_msg,
                            'activity_id': current.get('activity_id_', ''),
                            'process_key': current.get('process_key'),
                            'business_key': current.get('business_key'),
                            'hours_open': current.get('hours_open'),
                            'historical_occurrences': pattern.get('occurrence_count', 0),
                            'historical_resolution_rate': (
                                    pattern.get('resolved_count', 0) /
                                    pattern.get('occurrence_count', 1) * 100
                            ),
                            'root_cause': pattern.get('root_cause', 'Unknown'),
                            'priority': 'critical',
                            'similarity_score': round(similarity * 100, 1),
                            'recommendation': self._get_recurring_incident_recommendation(
                                pattern, current
                            )
                        })
                        break

        return recurring

    def _identify_repeated_runtime_incidents(self, current_incidents):
        """
        Fallback: Identify repeated incidents in runtime data only
        Used when history is disabled
        """
        # Group by type + message similarity
        groups = defaultdict(list)

        for inc in current_incidents:
            inc_type = inc.get('incident_type_', 'unknown')
            inc_msg = (inc.get('incident_msg_') or '')[:50]  # Shorter for grouping
            key = f"{inc_type}:{inc_msg}"
            groups[key].append(inc)

        # Find groups with multiple incidents (repeated)
        repeated = []
        for key, incidents in groups.items():
            if len(incidents) > 1:  # Repeated at least once
                # Take the oldest incident as the representative
                oldest = min(incidents, key=lambda x: x.get('incident_timestamp_', datetime.max))

                repeated.append({
                    'incident_id': oldest.get('incident_id'),
                    'incident_type': oldest.get('incident_type_', 'unknown'),
                    'error_message': oldest.get('incident_msg_', ''),
                    'activity_id': oldest.get('activity_id_', ''),
                    'process_key': oldest.get('process_key'),
                    'business_key': oldest.get('business_key'),
                    'hours_open': float(oldest.get('hours_open', 0)),
                    'historical_occurrences': len(incidents),  # Count in runtime
                    'historical_resolution_rate': 0,  # Unknown without history
                    'root_cause': self._categorize_incident_root_cause(
                        oldest.get('incident_type_', ''),
                        oldest.get('incident_msg_', '')
                    ),
                    'priority': 'high',
                    'similarity_score': 100,  # Exact match in runtime
                    'recommendation': f"This incident is currently repeating {len(incidents)} times. Investigate common cause immediately.",
                    'note': 'Detected in runtime only (history disabled)'
                })

        return repeated

    def _extract_root_causes(self, patterns):
        """Extract root cause categories from patterns"""
        root_cause_categories = defaultdict(int)
        for pattern in patterns:
            root_cause = pattern.get('root_cause', 'Unknown')
            root_cause_categories[root_cause] += pattern.get('occurrence_count', 0)
        return dict(root_cause_categories)

    def _get_runtime_incidents(self, lookback_days):
        """
        Get current runtime incidents with enhanced context
        IMPROVED: Better business key resolution and metadata
        """
        max_incidents = self._get_config('AI_MAX_INCIDENTS', 1000)

        query = f"""
            SELECT 
                inc.id_ AS incident_id,
                inc.incident_type_,
                inc.activity_id_,
                inc.incident_timestamp_,
                inc.incident_msg_,
                inc.proc_inst_id_,
                inc.failed_activity_id_,
                inc.execution_id_,
                pd.key_ as process_key,
                pd.name_ as process_name,
                COALESCE(
                    exec_main.business_key_, 
                    exec_parent.business_key_,
                    pi_hist.business_key_,
                    'N/A'
                ) AS business_key,
                EXTRACT(EPOCH FROM (NOW() - inc.incident_timestamp_))/3600 as hours_open,
                inc.job_def_id_,
                inc.cause_incident_id_,
                inc.root_cause_incident_id_,
                inc.configuration_,
                -- Get job details if it's a job incident
                job.retries_ as job_retries,
                job.exception_msg_ as job_exception_msg,
                -- Get activity name
                act.act_name_ as activity_name
            FROM act_ru_incident inc
            -- Main execution
            JOIN act_ru_execution exec_main ON exec_main.id_ = inc.execution_id_
            -- Process definition
            JOIN act_re_procdef pd ON exec_main.proc_def_id_ = pd.id_
            -- Parent execution (for business key fallback)
            LEFT JOIN act_ru_execution exec_parent ON exec_main.parent_id_ = exec_parent.id_
            -- Historical process instance (for business key fallback)
            LEFT JOIN act_hi_procinst pi_hist ON exec_main.proc_inst_id_ = pi_hist.proc_inst_id_
            -- Job details
            LEFT JOIN act_ru_job job ON inc.job_def_id_ = job.job_def_id_ 
                AND job.process_instance_id_ = inc.proc_inst_id_
            -- Activity instance for activity name
            LEFT JOIN act_hi_actinst act ON act.proc_inst_id_ = inc.proc_inst_id_ 
                AND act.act_id_ = inc.activity_id_
                AND act.end_time_ IS NULL
            WHERE inc.incident_timestamp_ > NOW() - INTERVAL '{lookback_days} days'
            ORDER BY inc.incident_timestamp_ ASC
            LIMIT {max_incidents}
        """

        return safe_execute(
            lambda: execute_query(query),
            default_value=[],
            context="Getting enhanced runtime incidents"
        )

    def _assess_incident_health(self, scored_incidents, historical_data):
        """
        Health assessment with granular levels
        """
        active_count = len(scored_incidents)
        critical_count = len([i for i in scored_incidents if i.get('severity') == 'critical'])
        high_count = len([i for i in scored_incidents if i.get('severity') == 'high'])

        # Get historical context
        total_historical = historical_data.get('total_incidents', 0)
        resolved_historical = historical_data.get('total_resolved', 0)
        resolution_rate = (resolved_historical / max(total_historical, 1)) * 100 if total_historical > 0 else 100

        # Get config thresholds
        critical_threshold = self._get_config('INCIDENT_HEALTH_CRITICAL_COUNT', 5)
        degraded_threshold = self._get_config('INCIDENT_HEALTH_DEGRADED_COUNT', 10)
        min_resolution = self._get_config('INCIDENT_HEALTH_MIN_RESOLUTION_RATE', 60)

        # Calculate age factor (incidents open > 48h are concerning)
        long_running = len([i for i in scored_incidents if i.get('age_hours', 0) > 48])

        # Determine health status with priority hierarchy
        if critical_count >= critical_threshold:
            return 'critical'
        elif long_running >= critical_threshold or active_count >= degraded_threshold * 2:
            return 'critical'
        elif high_count + critical_count >= degraded_threshold or resolution_rate < min_resolution:
            return 'degraded'
        elif active_count > 0 or resolution_rate < 80:
            return 'warning'
        else:
            return 'healthy'

    def _score_incident_severity(self, incidents):
        """
        Score incidents for priority (0-100)
        Factors: age, type, recurrence, process criticality
        """
        scored = []

        # Get config thresholds
        critical_age = self._get_config('INCIDENT_SEVERITY_CRITICAL_AGE_HOURS', 72)
        high_age = self._get_config('INCIDENT_SEVERITY_HIGH_AGE_HOURS', 24)
        medium_age = self._get_config('INCIDENT_SEVERITY_MEDIUM_AGE_HOURS', 12)
        warning_age = self._get_config('INCIDENT_SEVERITY_WARNING_AGE_HOURS', 4)

        for inc in incidents:
            hours_open = self._safe_float(inc.get('hours_open', 0)) or 0
            incident_type = inc.get('incident_type_', 'unknown')

            # Base severity by type (from config)
            type_severity_map = self._get_config('INCIDENT_TYPE_SEVERITY_MAP', {
                'failedJob': 50,
                'failedExternalTask': 45,
                'errorEventSubprocess': 40,
                'messageBoundaryEvent': 35,
                'timerStartEvent': 30
            })
            base_score = type_severity_map.get(incident_type, 40)

            # Age multiplier (incidents open longer are more severe)
            age_multiplier = 1.0
            if hours_open > critical_age:
                age_multiplier = 2.5
            elif hours_open > high_age:
                age_multiplier = 2.0
            elif hours_open > medium_age:
                age_multiplier = 1.5
            elif hours_open > warning_age:
                age_multiplier = 1.2

            # Calculate final severity score
            severity_score = min(100, base_score * age_multiplier)

            # Classify severity level (configurable thresholds)
            critical_threshold = self._get_config('INCIDENT_SEVERITY_CRITICAL_THRESHOLD', 80)
            high_threshold = self._get_config('INCIDENT_SEVERITY_HIGH_THRESHOLD', 60)
            medium_threshold = self._get_config('INCIDENT_SEVERITY_MEDIUM_THRESHOLD', 40)

            if severity_score >= critical_threshold:
                severity = 'critical'
            elif severity_score >= high_threshold:
                severity = 'high'
            elif severity_score >= medium_threshold:
                severity = 'medium'
            else:
                severity = 'low'

            scored.append({
                **inc,
                'severity_score': round(severity_score, 1),
                'severity': severity,
                'age_hours': round(hours_open, 1),
                'root_cause': self._categorize_incident_root_cause(
                    incident_type,
                    inc.get('incident_msg_', '')
                )
            })

        # Sort by severity score (highest first)
        scored.sort(key=lambda x: x['severity_score'], reverse=True)
        return scored

    def _identify_recurring_incidents(self, current_incidents, historical_patterns):
        """
        Identify incidents that appear in both current and historical
        These are high-priority as they indicate unresolved root causes
        """
        recurring = []

        # Get similarity threshold from config
        similarity_threshold = self._get_config('INCIDENT_SIMILARITY_THRESHOLD', 0.7)
        low_resolution_threshold = self._get_config('RECURRING_LOW_RESOLUTION_THRESHOLD', 50)

        for current in current_incidents:
            current_msg = (current.get('incident_msg_') or '')[:100]
            current_type = current.get('incident_type_', 'unknown')
            current_activity = current.get('activity_id_', '')

            # Find matching pattern in historical
            for pattern in historical_patterns:
                pattern_type = pattern.get('incident_type', '')
                pattern_msg = (pattern.get('error_message') or '')[:100]

                # Match if type is same and message is similar
                if pattern_type == current_type:
                    similarity = self._calculate_similarity(current_msg, pattern_msg)

                    if similarity > similarity_threshold:
                        recurring.append({
                            'incident_id': current.get('incident_id'),
                            'incident_type': current_type,
                            'error_message': current_msg,
                            'activity_id': current_activity,
                            'process_key': current.get('process_key'),
                            'business_key': current.get('business_key'),
                            'hours_open': current.get('hours_open'),
                            'historical_occurrences': pattern.get('occurrence_count', 0),
                            'historical_resolution_rate': (
                                    pattern.get('resolved_count', 0) /
                                    pattern.get('occurrence_count', 1) * 100
                            ),
                            'root_cause': pattern.get('root_cause', 'Unknown'),
                            'priority': 'critical',
                            'similarity_score': round(similarity * 100, 1),
                            'recommendation': self._get_recurring_incident_recommendation(
                                pattern, current
                            )
                        })
                        break

        return recurring

    def _calculate_similarity(self, str1, str2):
        """
        Similarity calculation using multiple techniques
        Combines Jaccard, n-gram, and keyword matching for better accuracy
        """
        if not str1 or not str2:
            return 0.0

        str1_lower = str1.lower()
        str2_lower = str2.lower()

        # 1. Exact match
        if str1_lower == str2_lower:
            return 1.0

        # 2. Substring containment (one contains the other)
        if str1_lower in str2_lower or str2_lower in str1_lower:
            shorter = min(len(str1_lower), len(str2_lower))
            longer = max(len(str1_lower), len(str2_lower))
            return 0.8 + (0.2 * (shorter / longer))

        # 3. Word-level Jaccard similarity
        words1 = set(str1_lower.split())
        words2 = set(str2_lower.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_score = len(intersection) / len(union) if union else 0.0

        # 4. Character n-gram similarity (trigrams)
        def get_ngrams(text, n=3):
            text = text.replace(' ', '')
            return set([text[i:i + n] for i in range(len(text) - n + 1)])

        ngrams1 = get_ngrams(str1_lower)
        ngrams2 = get_ngrams(str2_lower)

        if ngrams1 and ngrams2:
            ngram_intersection = ngrams1.intersection(ngrams2)
            ngram_union = ngrams1.union(ngrams2)
            ngram_score = len(ngram_intersection) / len(ngram_union) if ngram_union else 0.0
        else:
            ngram_score = 0.0

        # 5. Key technical terms matching (weighted heavily)
        # Common error keywords that are significant
        key_terms = {
            'timeout', 'connection', 'refused', 'deadlock', 'null', 'exception',
            'failed', 'error', 'invalid', 'unauthorized', 'forbidden', 'socket',
            'database', 'sql', 'jdbc', 'http', 'rest', 'api', 'network'
        }

        terms1 = words1.intersection(key_terms)
        terms2 = words2.intersection(key_terms)

        if terms1 or terms2:
            term_union = terms1.union(terms2)
            term_intersection = terms1.intersection(terms2)
            term_score = len(term_intersection) / len(term_union) if term_union else 0.0
        else:
            term_score = 0.0

        # 6. Weighted combination
        # Jaccard (40%), n-gram (30%), key terms (30%)
        final_score = (jaccard_score * 0.4) + (ngram_score * 0.3) + (term_score * 0.3)

        return min(1.0, final_score)

    def _generate_incident_recommendations(self, scored_incidents, recurring_issues, historical_data):
        """
        Granular and actionable recommendations
        """
        recommendations = []

        # 1. CRITICAL: Recurring incidents with low resolution
        if recurring_issues:
            critical_recurring = [r for r in recurring_issues
                                  if r.get('historical_resolution_rate', 100) < 50]
            if critical_recurring:
                # Group by root cause for targeted action
                by_root_cause = {}
                for issue in critical_recurring:
                    rc = issue.get('root_cause', 'Unknown')
                    if rc not in by_root_cause:
                        by_root_cause[rc] = []
                    by_root_cause[rc].append(issue)

                for root_cause, issues in by_root_cause.items():
                    recommendations.append({
                        'priority': 'critical',
                        'category': 'recurring_incidents',
                        'title': f'{len(issues)} Recurring {root_cause} Issues',
                        'message': f'Resolution rate < 50%. These keep happening without permanent fixes.',
                        'affected_processes': list(set([i['process_key'] for i in issues])),
                        'action': self._get_root_cause_action(root_cause),
                        'incident_ids': [i['incident_id'] for i in issues],
                        'details': {
                            'root_cause': root_cause,
                            'total_occurrences': sum(i.get('historical_occurrences', 0) for i in issues),
                            'avg_resolution_rate': sum(i.get('historical_resolution_rate', 0) for i in issues) / len(
                                issues)
                        }
                    })

        # 2. HIGH: Long-running incidents (>48h)
        very_old = [i for i in scored_incidents if i.get('age_hours', 0) > 48]
        if very_old:
            recommendations.append({
                'priority': 'high',
                'category': 'aging_incidents',
                'title': f'{len(very_old)} Incidents Open > 48 Hours',
                'message': f'Oldest: {max([i.get("age_hours", 0) for i in very_old]):.1f}h - Immediate resolution needed',
                'action': 'Escalate to senior engineers and prioritize resolution',
                'incident_ids': [i['incident_id'] for i in very_old][:10],  # Top 10
                'details': {
                    'oldest_age': max([i.get("age_hours", 0) for i in very_old]),
                    'by_severity': {
                        'critical': len([i for i in very_old if i.get('severity') == 'critical']),
                        'high': len([i for i in very_old if i.get('severity') == 'high'])
                    }
                }
            })

        # 3. MEDIUM: Aging incidents (24-48h)
        aging = [i for i in scored_incidents if 24 <= i.get('age_hours', 0) <= 48]
        if aging:
            recommendations.append({
                'priority': 'medium',
                'category': 'aging_incidents',
                'title': f'{len(aging)} Incidents Approaching Critical Age (24-48h)',
                'message': 'Review and resolve before they become critical',
                'action': 'Assign owners and set resolution deadlines',
                'incident_ids': [i['incident_id'] for i in aging][:10]
            })

        # 4. Group by root cause for pattern detection
        root_cause_counts = {}
        for inc in scored_incidents:
            cause = inc.get('root_cause', 'Unknown')
            if cause not in root_cause_counts:
                root_cause_counts[cause] = {'count': 0, 'severities': []}
            root_cause_counts[cause]['count'] += 1
            root_cause_counts[cause]['severities'].append(inc.get('severity', 'low'))

        # Find dominant root cause
        if root_cause_counts:
            top_cause = max(root_cause_counts.items(), key=lambda x: x[1]['count'])
            if top_cause[1]['count'] >= 3:
                critical_in_cause = top_cause[1]['severities'].count('critical')
                priority = 'high' if critical_in_cause > 0 else 'medium'

                recommendations.append({
                    'priority': priority,
                    'category': 'root_cause_pattern',
                    'title': f'Pattern Detected: {top_cause[0]} ({top_cause[1]["count"]} incidents)',
                    'message': f'{critical_in_cause} critical incidents share this root cause',
                    'action': self._get_root_cause_action(top_cause[0]),
                    'details': {
                        'root_cause': top_cause[0],
                        'count': top_cause[1]['count'],
                        'severity_breakdown': {
                            'critical': top_cause[1]['severities'].count('critical'),
                            'high': top_cause[1]['severities'].count('high'),
                            'medium': top_cause[1]['severities'].count('medium'),
                            'low': top_cause[1]['severities'].count('low')
                        }
                    }
                })

        # 5. Check incident creation rate
        if historical_data.get('total_incidents', 0) > 0:
            lookback = historical_data.get('analysis_window_days', 30)
            daily_rate = historical_data['total_incidents'] / max(lookback, 1)

            if daily_rate > 10:
                recommendations.append({
                    'priority': 'high',
                    'category': 'incident_rate',
                    'title': f'High Incident Creation Rate: {daily_rate:.1f}/day',
                    'message': 'System generating incidents faster than resolution capacity',
                    'action': 'Implement preventive measures: better error handling, input validation, retry mechanisms',
                    'details': {
                        'daily_rate': round(daily_rate, 2),
                        'weekly_rate': round(daily_rate * 7, 1),
                        'trend': 'increasing' if daily_rate > 5 else 'stable'
                    }
                })
            elif daily_rate > 5:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'incident_rate',
                    'title': f'Elevated Incident Rate: {daily_rate:.1f}/day',
                    'message': 'Monitor trend - may need preventive action if rate increases',
                    'action': 'Review error patterns and implement automated recovery where possible'
                })

        # 6. Process-specific recommendations
        process_incidents = {}
        for inc in scored_incidents:
            proc = inc.get('process_key', 'unknown')
            if proc not in process_incidents:
                process_incidents[proc] = []
            process_incidents[proc].append(inc)

        # Find processes with multiple critical incidents
        for process_key, incidents in process_incidents.items():
            critical_count = len([i for i in incidents if i.get('severity') == 'critical'])
            if critical_count >= 3:
                recommendations.append({
                    'priority': 'high',
                    'category': 'process_health',
                    'title': f'Process at Risk: {process_key}',
                    'message': f'{critical_count} critical incidents in this process',
                    'action': f'Review process definition, error boundaries, and external dependencies for {process_key}',
                    'affected_processes': [process_key],
                    'details': {
                        'total_incidents': len(incidents),
                        'critical': critical_count,
                        'avg_age_hours': sum(i.get('age_hours', 0) for i in incidents) / len(incidents)
                    }
                })

        # Default if no recommendations
        if not recommendations:
            recommendations.append({
                'priority': 'low',
                'category': 'general',
                'message': 'No critical issues detected - Continue standard monitoring',
                'action': 'Maintain current incident response procedures'
            })

        return recommendations

    def _get_recurring_incident_recommendation(self, pattern, current):
        """Get specific recommendation for recurring incident"""
        resolution_rate = (
                pattern.get('resolved_count', 0) /
                pattern.get('occurrence_count', 1) * 100
        )

        if resolution_rate < 30:
            return f"CRITICAL: This incident has occurred {pattern.get('occurrence_count', 0)} times with only {resolution_rate:.0f}% resolution rate. Implement permanent fix immediately."
        elif resolution_rate < 70:
            return f"WARNING: Recurring incident ({pattern.get('occurrence_count', 0)}x) with {resolution_rate:.0f}% resolution. Review error handling and add retry logic."
        else:
            return f"This incident pattern has been seen {pattern.get('occurrence_count', 0)} times but usually resolves. Monitor for changes."

    def _get_root_cause_action(self, root_cause):
        """Get recommended action for root cause category"""
        actions = {
            'Database': 'Review database connection pool settings, query performance, and connection timeout configurations',
            'External Service': 'Implement circuit breakers, increase timeouts, add retry logic with exponential backoff',
            'Business Logic': 'Review validation rules, add defensive coding, improve error messages',
            'Configuration': 'Audit configuration files, implement configuration validation, use environment-specific configs',
            'Resource': 'Scale infrastructure, optimize memory usage, review thread pool configurations',
            'Authorization': 'Review security policies, check service account permissions, audit access controls',
            'Timeout': 'Increase timeout thresholds, implement async processing, add progress indicators',
            'Job Execution': 'Review retry configuration, increase job priority, optimize job execution'
        }
        return actions.get(root_cause, 'Investigate root cause and implement appropriate fixes')

    def _get_critical_alerts(self, scored_incidents, recurring_issues):
        """Get list of critical alerts requiring immediate attention"""
        alerts = []

        # Critical severity incidents
        critical = [i for i in scored_incidents if i.get('severity') == 'critical']
        if critical:
            alerts.append({
                'type': 'critical_severity',
                'count': len(critical),
                'message': f'{len(critical)} critical severity incidents require immediate attention',
                'incidents': critical[:5]  # Top 5
            })

        # Recurring issues
        if recurring_issues:
            alerts.append({
                'type': 'recurring',
                'count': len(recurring_issues),
                'message': f'{len(recurring_issues)} recurring incidents indicate systemic issues',
                'incidents': recurring_issues[:5]
            })

        return alerts

    def _assess_incident_health(self, scored_incidents, historical_patterns):
        """Assess overall incident health status"""
        active_count = len(scored_incidents)
        critical_count = len([i for i in scored_incidents if i.get('severity') == 'critical'])

        total_historical = historical_patterns.get('total_incidents', 0)
        resolved_historical = historical_patterns.get('total_resolved', 0)
        resolution_rate = (resolved_historical / max(total_historical, 1)) * 100

        if critical_count > 5:
            return 'critical'
        elif active_count > 10 or resolution_rate < 60:
            return 'degraded'
        elif active_count > 0:
            return 'warning'
        else:
            return 'healthy'

    def _calculate_resolution_rate(self, historical_patterns):
        """Calculate resolution rate from historical patterns"""
        total = historical_patterns.get('total_incidents', 0)
        resolved = historical_patterns.get('total_resolved', 0)
        return round((resolved / max(total, 1)) * 100, 1)

    def _calculate_avg_resolution_time(self, historical_patterns):
        """Calculate average resolution time from patterns"""
        patterns = historical_patterns.get('patterns', [])
        if not patterns:
            return 0

        total_hours = 0
        resolved_count = 0

        for pattern in patterns:
            resolved = pattern.get('resolved_count', 0)
            avg_duration = pattern.get('avg_duration_hours', 0)

            if resolved > 0 and avg_duration > 0:
                total_hours += avg_duration * resolved
                resolved_count += resolved

        return round(total_hours / max(resolved_count, 1), 2)

    def _group_by_severity(self, incidents):
        """Group incidents by severity level"""
        grouped = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for inc in incidents:
            severity = inc.get('severity', 'low')
            grouped[severity] = grouped.get(severity, 0) + 1
        return grouped

    def _group_by_process(self, incidents):
        """Group incidents by process definition"""
        grouped = {}
        for inc in incidents:
            proc = inc.get('process_key', 'unknown')
            if proc not in grouped:
                grouped[proc] = {'count': 0, 'incidents': []}
            grouped[proc]['count'] += 1
            grouped[proc]['incidents'].append(inc['incident_id'])

        # Return top 10 processes
        return dict(sorted(grouped.items(), key=lambda x: x[1]['count'], reverse=True)[:10])

    def _get_oldest_incident_age(self, incidents):
        """Get age of oldest incident in hours"""
        if not incidents:
            return 0
        ages = [self._safe_float(i.get('hours_open', 0)) for i in incidents]
        return round(max(ages) if ages else 0, 1)

    @timed_cache(seconds=3600)
    def analyze_incident_patterns(self, lookback_days=None):
        """
        Cluster similar incidents using text analysis
        incl. business keys, status tracking, and sample instances
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
                            inc.id_ as incident_id,
                            inc.incident_type_,
                            inc.incident_msg_,
                            inc.proc_def_key_,
                            inc.activity_id_,
                            inc.create_time_,
                            inc.end_time_,
                            inc.proc_inst_id_,
                            pi.business_key_,
                            CASE
                                WHEN inc.end_time_ IS NULL THEN 'open'
                                WHEN inc.incident_state_ = 2 THEN 'resolved'
                                WHEN inc.removal_time_ IS NOT NULL THEN 'deleted'
                                ELSE 'resolved'
                            END as status,
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
    # EXTREME VARIABILITY DETECTION
    # =========================================================================

    @timed_cache(seconds=3600)
    def analyze_extreme_variability(self, process_categories=None):
        """
        Detect processes with extreme P95/Median ratios (dangerously unpredictable)
        Includes sample slow instances with business keys
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

                    # Get sample slow instances
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

    # =========================================================================
    # ALIAS FUNCTIONS (for API compatibility)
    # =========================================================================

    @timed_cache(seconds=3600)
    def detect_extreme_variability(self, lookback_days=None):
        """
        Alias for analyze_extreme_variability (for API endpoint compatibility)
        """
        # Get process categories first
        cat_result = self.get_process_categories(lookback_days=lookback_days)
        process_categories = cat_result.get('categories', {})

        # Call the main analysis function
        return self.analyze_extreme_variability(process_categories=process_categories)

    def _get_variability_recommendation(self, ratio: float, severity: str) -> str:
        """Get recommendation for processes with extreme variability"""
        if severity == 'extreme':
            return f'CRITICAL: P95 is {ratio:.0f}x median - Do NOT use for SLA-critical workflows. Investigate external dependencies, stuck subprocesses, or consider splitting into fast/slow paths.'
        elif severity == 'high':
            return f'WARNING: P95 is {ratio:.0f}x median - Implement timeout safeguards and add monitoring alerts for P99 violations.'
        else:
            return f'CAUTION: P95 is {ratio:.0f}x median - Monitor closely and investigate outlier patterns.'

    # =========================================================================
    # STUCK PROCESS DETECTION
    # =========================================================================

    @timed_cache(seconds=3600)
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

    @timed_cache(seconds=3600)
    def analyze_outlier_patterns(self, lookback_days=None):
        """
        IQR-based outlier detection for each process
        Includes extreme outlier tracking, sample instances, and process category filtering
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

                # Get sample outlier instances if high severity
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
        Generate critical insights summary
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
    # BOTTLENECK IDENTIFICATION
    # =========================================================================

    @timed_cache(seconds=3600)
    def identify_bottlenecks(self, lookback_days=None):
        """
        Identify process bottlenecks by analyzing activity durations
        incl. activity types, CV, and specific recommendations
        Filters to business-critical processes only for better performance
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

    @timed_cache(seconds=3600)
    def predict_job_failures(self, lookback_days=None):
        """
        Analyze job failure patterns and predict failure-prone jobs
        Filters to business-critical processes only for better performance
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

    @timed_cache(seconds=300)
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

    @timed_cache(seconds=3600)
    def get_process_leaderboard(self, lookback_days=None):
        """
        Performance leaderboard for process definitions
        Filters to business-critical processes only for better performance
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

    @timed_cache(seconds=3600)
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



# Singleton instance
_ai_analytics = AIAnalytics()


def get_ai_analytics():
    """Get the AI analytics singleton"""
    return _ai_analytics


