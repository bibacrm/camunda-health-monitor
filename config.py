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

