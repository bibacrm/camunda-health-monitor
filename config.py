"""
Configuration module for Camunda Health Monitor
All configuration from environment variables
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration"""

    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    PORT = int(os.getenv('PORT', '5000'))
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

    # Logging Configuration
    JSON_LOGGING = os.getenv('JSON_LOGGING', 'false').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # Database Configuration
    DB_CONFIG = {
        'dbname': os.getenv('DB_NAME', 'camunda'),
        'user': os.getenv('DB_USER', 'camunda'),
        'password': os.getenv('DB_PASSWORD', 'camunda'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'connect_timeout': 5,  # Fail fast - 5 seconds max
        'options': '-c statement_timeout=60000',  # 60 second query timeout
        'keepalives': 1,
        'keepalives_idle': 5,
        'keepalives_interval': 2,
        'keepalives_count': 2
    }

    # SSL Verification
    SSL_VERIFY = os.getenv('SSL_VERIFY', 'false').lower() == 'true'

    # Application Settings
    STUCK_INSTANCE_DAYS = int(os.getenv('STUCK_INSTANCE_DAYS', '7'))
    JVM_METRICS_SOURCE = os.getenv('JVM_METRICS_SOURCE', 'jmx')

    # ========================================================================
    # AI/ML Configuration
    # ========================================================================

    # Edition: 'oss' or 'enterprise'
    EDITION = os.getenv('EDITION', 'oss')

    # Analysis Time Window
    # --------------------
    # How many days back to analyze for all AI insights
    AI_LOOKBACK_DAYS = int(os.getenv('AI_LOOKBACK_DAYS', '30'))

    # Query Limits
    # ------------
    # Maximum instances to fetch for AI analysis (process, activity, job instances)
    AI_MAX_INSTANCES = int(os.getenv('AI_MAX_INSTANCES', '50000'))

    # Maximum incidents to fetch for pattern recognition
    AI_MAX_INCIDENTS = int(os.getenv('AI_MAX_INCIDENTS', '1000'))

    # Minimum Data Requirements
    # --------------------------
    # Minimum data points needed for AI analysis (instances, executions, etc.)
    AI_MIN_DATA = int(os.getenv('AI_MIN_DATA', '10'))

    # Detection Thresholds
    # --------------------
    # Z-score threshold for statistical anomaly detection (lower = more sensitive)
    AI_ZSCORE_THRESHOLD = float(os.getenv('AI_ZSCORE_THRESHOLD', '1.0'))

    # Maximum execution time in seconds to flag as critical stuck process
    AI_CRITICAL_DURATION_SECONDS = int(os.getenv('AI_CRITICAL_DURATION_SECONDS', '86400'))  # 1 day

    # Maximum execution time in seconds to flag as high severity
    AI_HIGH_DURATION_SECONDS = int(os.getenv('AI_HIGH_DURATION_SECONDS', '7200'))  # 2 hours

    # Maximum execution time in seconds to flag as medium severity
    AI_MEDIUM_DURATION_SECONDS = int(os.getenv('AI_MEDIUM_DURATION_SECONDS', '3600'))  # 1 hour

    # Average execution time in seconds to flag as slow
    AI_SLOW_AVG_DURATION_SECONDS = int(os.getenv('AI_SLOW_AVG_DURATION_SECONDS', '600'))  # 10 minutes

    # SLA Configuration
    # -----------------
    # Default SLA threshold in hours for breach prediction
    SLA_THRESHOLD_HOURS = int(os.getenv('SLA_THRESHOLD_HOURS', '24'))

    # Percentage of SLA threshold to start flagging tasks as at-risk
    SLA_WARNING_THRESHOLD_PCT = int(os.getenv('SLA_WARNING_THRESHOLD_PCT', '70'))

    # Result Limits (UI Display)
    # --------------------------
    # Maximum number of results to return in API responses and display in UI
    AI_UI_RESULTS_LIMIT = int(os.getenv('AI_UI_RESULTS_LIMIT', '20'))

    # UI/Frontend Configuration
    # -------------------------
    # Auto-refresh interval in milliseconds (default: 30 seconds)
    UI_AUTO_REFRESH_INTERVAL_MS = int(os.getenv('UI_AUTO_REFRESH_INTERVAL_MS', '30000'))

    # Archive threshold in days - instances older than this are considered archivable
    DB_ARCHIVE_THRESHOLD_DAYS = int(os.getenv('DB_ARCHIVE_THRESHOLD_DAYS', '90'))

    # Advanced ML Features
    # --------------------
    # Capacity forecasting: days to forecast ahead
    AI_CAPACITY_FORECAST_DAYS = int(os.getenv('AI_CAPACITY_FORECAST_DAYS', '30'))

    # Capacity forecasting: training data days
    AI_CAPACITY_TRAINING_DAYS = int(os.getenv('AI_CAPACITY_TRAINING_DAYS', '90'))

    # Variable impact: minimum impact percentage to report
    AI_MIN_VARIABLE_IMPACT_PCT = int(os.getenv('AI_MIN_VARIABLE_IMPACT_PCT', '10'))

    # Stuck activity: percentile threshold for stuck detection (P95 = 95th percentile)
    AI_STUCK_ACTIVITY_PERCENTILE = int(os.getenv('AI_STUCK_ACTIVITY_PERCENTILE', '95'))

    # Stuck activity: multiplier for stuck threshold (activity > P95 * multiplier)
    AI_STUCK_ACTIVITY_MULTIPLIER = float(os.getenv('AI_STUCK_ACTIVITY_MULTIPLIER', '2.0'))

    # Duration prediction: minimum confidence level to show predictions
    AI_DURATION_PREDICTION_MIN_CONFIDENCE = float(os.getenv('AI_DURATION_PREDICTION_MIN_CONFIDENCE', '0.7'))

    # Duration prediction: minimum training instances needed
    AI_DURATION_PREDICTION_MIN_TRAINING = int(os.getenv('AI_DURATION_PREDICTION_MIN_TRAINING', '50'))

    # Process Categorization
    # ----------------------
    # Process category thresholds (in hours) - used to classify process duration
    PROCESS_CATEGORY_THRESHOLDS = {
        'ultra_fast': 5 / 3600,      # < 5 seconds
        'very_fast': 0.5 / 60,        # 5-30 seconds
        'fast_background': 0.1,       # 30s - 6 minutes
        'standard': 0.5,              # 6 - 30 minutes
        'extended': 4,                # 30 min - 4 hours
        'long_running': 24,           # 4 - 24 hours
        'batch_manual': float('inf')  # > 24 hours
    }

    # Process category labels for UI display
    PROCESS_CATEGORY_LABELS = {
        'ultra_fast': 'Ultra Fast (<5s)',
        'very_fast': 'Very Fast (5-30s)',
        'fast_background': 'Fast Background (<6m)',
        'standard': 'Standard (6m-30m)',
        'extended': 'Extended (30m-4h)',
        'long_running': 'Long Running (4h-24h)',
        'batch_manual': 'Batch / Manual (24h+)'
    }

    # Categories to analyze as business-critical (exclude ultra_fast for performance)
    ANALYZE_CATEGORIES = ['very_fast', 'fast_background', 'standard', 'extended', 'long_running', 'batch_manual']

    # Version Performance Analysis
    # ----------------------------
    # Percentage threshold for flagging version regressions
    VERSION_REGRESSION_THRESHOLD_PCT = float(os.getenv('VERSION_REGRESSION_THRESHOLD_PCT', '20.0'))

    # Load Pattern Analysis
    # ---------------------
    # Business hours definition for load pattern analysis
    BUSINESS_HOURS_START = int(os.getenv('BUSINESS_HOURS_START', '7'))   # 7 AM
    BUSINESS_HOURS_END = int(os.getenv('BUSINESS_HOURS_END', '19'))      # 7 PM

    # Weekend days (0=Sunday, 6=Saturday in PostgreSQL EXTRACT(DOW))
    WEEKEND_DAYS = [0, 6]

    # Stuck Process Detection
    # -----------------------
    # P95 multipliers for stuck process severity classification
    STUCK_PROCESS_P95_MULTIPLIER_CRITICAL = float(os.getenv('STUCK_PROCESS_P95_MULTIPLIER_CRITICAL', '3.0'))
    STUCK_PROCESS_P95_MULTIPLIER_WARNING = float(os.getenv('STUCK_PROCESS_P95_MULTIPLIER_WARNING', '2.0'))
    STUCK_PROCESS_P95_MULTIPLIER_ATTENTION = float(os.getenv('STUCK_PROCESS_P95_MULTIPLIER_ATTENTION', '1.5'))

    # Extreme Variability Detection
    # ------------------------------
    # P95/Median ratio thresholds for variability classification
    EXTREME_VARIABILITY_RATIO_EXTREME = float(os.getenv('EXTREME_VARIABILITY_RATIO_EXTREME', '100.0'))
    EXTREME_VARIABILITY_RATIO_HIGH = float(os.getenv('EXTREME_VARIABILITY_RATIO_HIGH', '50.0'))
    EXTREME_VARIABILITY_RATIO_MEDIUM = float(os.getenv('EXTREME_VARIABILITY_RATIO_MEDIUM', '20.0'))

    # Stability Classification (Coefficient of Variation)
    # ----------------------------------------------------
    # CV thresholds for classifying process stability
    STABILITY_CV_STABLE_THRESHOLD = float(os.getenv('STABILITY_CV_STABLE_THRESHOLD', '0.3'))
    STABILITY_CV_MODERATE_THRESHOLD = float(os.getenv('STABILITY_CV_MODERATE_THRESHOLD', '1.0'))

    # Database Health Thresholds
    # --------------------------
    # Latency thresholds in milliseconds for DB health scoring
    DB_LATENCY_EXCELLENT_MS = int(os.getenv('DB_LATENCY_EXCELLENT_MS', '10'))
    DB_LATENCY_GOOD_MS = int(os.getenv('DB_LATENCY_GOOD_MS', '50'))
    DB_LATENCY_FAIR_MS = int(os.getenv('DB_LATENCY_FAIR_MS', '100'))
    DB_LATENCY_POOR_MS = int(os.getenv('DB_LATENCY_POOR_MS', '500'))

    # Incident Rate Thresholds (percentage)
    # --------------------------------------
    INCIDENT_RATE_EXCELLENT_PCT = float(os.getenv('INCIDENT_RATE_EXCELLENT_PCT', '1.0'))
    INCIDENT_RATE_GOOD_PCT = float(os.getenv('INCIDENT_RATE_GOOD_PCT', '5.0'))
    INCIDENT_RATE_FAIR_PCT = float(os.getenv('INCIDENT_RATE_FAIR_PCT', '10.0'))

    # Health Factor Alert Thresholds
    # -------------------------------
    # Thresholds for when health factors trigger warnings
    HEALTH_JVM_WARNING_THRESHOLD = int(os.getenv('HEALTH_JVM_WARNING_THRESHOLD', '70'))
    HEALTH_DB_WARNING_THRESHOLD = int(os.getenv('HEALTH_DB_WARNING_THRESHOLD', '70'))
    HEALTH_INCIDENTS_WARNING_THRESHOLD = int(os.getenv('HEALTH_INCIDENTS_WARNING_THRESHOLD', '80'))
    HEALTH_JOBS_WARNING_THRESHOLD = int(os.getenv('HEALTH_JOBS_WARNING_THRESHOLD', '90'))

    # Version Regression Severity Thresholds
    # ---------------------------------------
    VERSION_REGRESSION_HIGH_PCT = float(os.getenv('VERSION_REGRESSION_HIGH_PCT', '50.0'))
    VERSION_REGRESSION_CRITICAL_PCT = float(os.getenv('VERSION_REGRESSION_CRITICAL_PCT', '100.0'))

    # Load Pattern Thresholds
    # ------------------------
    LOAD_PATTERN_WEEKEND_LOW_THRESHOLD_PCT = float(os.getenv('LOAD_PATTERN_WEEKEND_LOW_THRESHOLD_PCT', '20.0'))
    LOAD_PATTERN_WEEKEND_SIMILAR_THRESHOLD = float(os.getenv('LOAD_PATTERN_WEEKEND_SIMILAR_THRESHOLD', '0.9'))
    LOAD_PATTERN_WEEKEND_SLOWER_THRESHOLD = float(os.getenv('LOAD_PATTERN_WEEKEND_SLOWER_THRESHOLD', '1.1'))

    # Outlier Detection Thresholds
    # -----------------------------
    # IQR multipliers for outlier detection
    OUTLIER_IQR_MULTIPLIER_NORMAL = float(os.getenv('OUTLIER_IQR_MULTIPLIER_NORMAL', '1.5'))
    OUTLIER_IQR_MULTIPLIER_EXTREME = float(os.getenv('OUTLIER_IQR_MULTIPLIER_EXTREME', '3.0'))
    OUTLIER_HIGH_PERCENTAGE_THRESHOLD = float(os.getenv('OUTLIER_HIGH_PERCENTAGE_THRESHOLD', '15.0'))
    OUTLIER_MEDIUM_PERCENTAGE_THRESHOLD = float(os.getenv('OUTLIER_MEDIUM_PERCENTAGE_THRESHOLD', '5.0'))

    # Anomaly Detection Thresholds
    # -----------------------------
    # Z-score thresholds for anomaly severity classification
    ANOMALY_ZSCORE_HIGH_THRESHOLD = float(os.getenv('ANOMALY_ZSCORE_HIGH_THRESHOLD', '3.0'))
    ANOMALY_ZSCORE_MEDIUM_THRESHOLD = float(os.getenv('ANOMALY_ZSCORE_MEDIUM_THRESHOLD', '2.0'))

    # Job Failure Rate Thresholds (percentage)
    # -----------------------------------------
    JOB_FAILURE_CRITICAL_PCT = float(os.getenv('JOB_FAILURE_CRITICAL_PCT', '20.0'))
    JOB_FAILURE_WARNING_PCT = float(os.getenv('JOB_FAILURE_WARNING_PCT', '10.0'))
    JOB_FAILURE_MONITOR_PCT = float(os.getenv('JOB_FAILURE_MONITOR_PCT', '5.0'))

    # Capacity Forecasting Thresholds
    # --------------------------------
    CAPACITY_TREND_SIGNIFICANT_THRESHOLD = float(os.getenv('CAPACITY_TREND_SIGNIFICANT_THRESHOLD', '0.2'))
    CAPACITY_TREND_STABLE_THRESHOLD = float(os.getenv('CAPACITY_TREND_STABLE_THRESHOLD', '1.0'))

    # Process Performance Grading
    # ----------------------------
    # Thresholds for process performance grades
    PROCESS_GRADE_A_COMPLETION_PCT = float(os.getenv('PROCESS_GRADE_A_COMPLETION_PCT', '95.0'))
    PROCESS_GRADE_A_FAILURE_PCT = float(os.getenv('PROCESS_GRADE_A_FAILURE_PCT', '1.0'))
    PROCESS_GRADE_B_COMPLETION_PCT = float(os.getenv('PROCESS_GRADE_B_COMPLETION_PCT', '90.0'))
    PROCESS_GRADE_B_FAILURE_PCT = float(os.getenv('PROCESS_GRADE_B_FAILURE_PCT', '5.0'))
    PROCESS_GRADE_C_COMPLETION_PCT = float(os.getenv('PROCESS_GRADE_C_COMPLETION_PCT', '80.0'))
    PROCESS_GRADE_C_FAILURE_PCT = float(os.getenv('PROCESS_GRADE_C_FAILURE_PCT', '10.0'))
    PROCESS_GRADE_D_COMPLETION_PCT = float(os.getenv('PROCESS_GRADE_D_COMPLETION_PCT', '70.0'))

    # Node Performance Thresholds
    # ----------------------------
    NODE_PERFORMANCE_EXCELLENT_THRESHOLD = int(os.getenv('NODE_PERFORMANCE_EXCELLENT_THRESHOLD', '80'))
    NODE_PERFORMANCE_GOOD_THRESHOLD = int(os.getenv('NODE_PERFORMANCE_GOOD_THRESHOLD', '60'))
    NODE_PERFORMANCE_POOR_THRESHOLD = int(os.getenv('NODE_PERFORMANCE_POOR_THRESHOLD', '40'))

    # Activity Duration Thresholds
    # -----------------------------
    # Minimum average duration in seconds to consider activity as potential bottleneck
    BOTTLENECK_MIN_DURATION_SECONDS = float(os.getenv('BOTTLENECK_MIN_DURATION_SECONDS', '1.0'))

    # Sample Instance Limits
    # ----------------------
    INCIDENT_PATTERN_SAMPLE_LIMIT = int(os.getenv('INCIDENT_PATTERN_SAMPLE_LIMIT', '5'))
    LOAD_PATTERN_PEAK_HOURS_LIMIT = int(os.getenv('LOAD_PATTERN_PEAK_HOURS_LIMIT', '10'))

    # Stuck Activity Detection
    # ------------------------
    # Fallback threshold for stuck detection when no historical data (in seconds, 24h)
    STUCK_ACTIVITY_FALLBACK_THRESHOLD_SECONDS = int(os.getenv('STUCK_ACTIVITY_FALLBACK_THRESHOLD_SECONDS', '86400'))

    # ========================================================================
    # Enhanced Incident Analysis Configuration
    # ========================================================================

    # Incident Severity Scoring
    # --------------------------
    # Age thresholds for severity classification (in hours)
    INCIDENT_SEVERITY_CRITICAL_AGE_HOURS = int(os.getenv('INCIDENT_SEVERITY_CRITICAL_AGE_HOURS', '72'))  # 3 days
    INCIDENT_SEVERITY_HIGH_AGE_HOURS = int(os.getenv('INCIDENT_SEVERITY_HIGH_AGE_HOURS', '24'))  # 1 day
    INCIDENT_SEVERITY_MEDIUM_AGE_HOURS = int(os.getenv('INCIDENT_SEVERITY_MEDIUM_AGE_HOURS', '12'))  # 12 hours
    INCIDENT_SEVERITY_WARNING_AGE_HOURS = int(os.getenv('INCIDENT_SEVERITY_WARNING_AGE_HOURS', '4'))  # 4 hours

    # Severity score thresholds (0-100)
    INCIDENT_SEVERITY_CRITICAL_THRESHOLD = int(os.getenv('INCIDENT_SEVERITY_CRITICAL_THRESHOLD', '80'))
    INCIDENT_SEVERITY_HIGH_THRESHOLD = int(os.getenv('INCIDENT_SEVERITY_HIGH_THRESHOLD', '60'))
    INCIDENT_SEVERITY_MEDIUM_THRESHOLD = int(os.getenv('INCIDENT_SEVERITY_MEDIUM_THRESHOLD', '40'))

    # Base severity scores by incident type
    INCIDENT_TYPE_SEVERITY_MAP = {
        'failedJob': int(os.getenv('INCIDENT_TYPE_FAILED_JOB_SEVERITY', '50')),
        'failedExternalTask': int(os.getenv('INCIDENT_TYPE_EXTERNAL_TASK_SEVERITY', '45')),
        'errorEventSubprocess': int(os.getenv('INCIDENT_TYPE_ERROR_EVENT_SEVERITY', '40')),
        'messageBoundaryEvent': int(os.getenv('INCIDENT_TYPE_MESSAGE_BOUNDARY_SEVERITY', '35')),
        'timerStartEvent': int(os.getenv('INCIDENT_TYPE_TIMER_START_SEVERITY', '30'))
    }

    # Recurring Incident Detection
    # -----------------------------
    # Similarity threshold for matching incidents (0.0-1.0, 0.7 = 70% match)
    INCIDENT_SIMILARITY_THRESHOLD = float(os.getenv('INCIDENT_SIMILARITY_THRESHOLD', '0.7'))

    # Resolution rate threshold for flagging as problematic (percentage)
    RECURRING_LOW_RESOLUTION_THRESHOLD = int(os.getenv('RECURRING_LOW_RESOLUTION_THRESHOLD', '50'))

    # Incident Health Assessment
    # ---------------------------
    # Number of critical incidents to trigger critical health status
    INCIDENT_HEALTH_CRITICAL_COUNT = int(os.getenv('INCIDENT_HEALTH_CRITICAL_COUNT', '5'))

    # Active incident count to trigger degraded status
    INCIDENT_HEALTH_DEGRADED_COUNT = int(os.getenv('INCIDENT_HEALTH_DEGRADED_COUNT', '10'))

    # Minimum resolution rate for healthy status (percentage)
    INCIDENT_HEALTH_MIN_RESOLUTION_RATE = int(os.getenv('INCIDENT_HEALTH_MIN_RESOLUTION_RATE', '60'))

    # Timeline Visualization
    # ----------------------
    # Default days for timeline view
    INCIDENT_TIMELINE_DEFAULT_DAYS = int(os.getenv('INCIDENT_TIMELINE_DEFAULT_DAYS', '7'))

    # Time bucket granularity (hour, day, week)
    INCIDENT_TIMELINE_BUCKET_SIZE = os.getenv('INCIDENT_TIMELINE_BUCKET_SIZE', 'hour')

    # Maximum data points in timeline chart
    INCIDENT_TIMELINE_MAX_POINTS = int(os.getenv('INCIDENT_TIMELINE_MAX_POINTS', '168'))  # 7 days * 24 hours


    @staticmethod
    def load_camunda_nodes():
        """Load Camunda nodes from environment"""
        nodes = {}
        node_index = 1

        while True:
            node_name = os.getenv(f'CAMUNDA_NODE_{node_index}_NAME')
            node_url = os.getenv(f'CAMUNDA_NODE_{node_index}_URL')
            if not node_name or not node_url:
                break
            nodes[node_name] = node_url
            node_index += 1

        # If no nodes configured, use defaults
        if not nodes:
            nodes = {
                'node1': os.getenv('CAMUNDA_URL', 'http://localhost:8080/engine-rest')
            }

        return nodes

    @staticmethod
    def load_jmx_endpoints():
        """Load JMX endpoints from environment"""
        endpoints = {}
        node_index = 1

        while True:
            jmx_url = os.getenv(f'JMX_NODE_{node_index}_URL')
            node_name = os.getenv(f'CAMUNDA_NODE_{node_index}_NAME', f'node{node_index}')
            if not jmx_url:
                break
            endpoints[node_name] = jmx_url
            node_index += 1

        return endpoints

    @staticmethod
    def get_camunda_auth():
        """Get Camunda API authentication"""
        user = os.getenv('CAMUNDA_API_USER')
        password = os.getenv('CAMUNDA_API_PASSWORD')
        if user and password:
            return (user, password)
        return None

    @staticmethod
    def validate():
        """Validate required configuration"""
        required = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing = [var for var in required if not os.getenv(var)]

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    JSON_LOGGING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    JSON_LOGGING = True


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# Configuration mapping
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """Get configuration by name"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    return config_by_name.get(config_name, DevelopmentConfig)

