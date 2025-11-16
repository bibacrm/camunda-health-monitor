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
        'options': '-c statement_timeout=30000',  # 30 second query timeout
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

