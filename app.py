"""
Camunda 7 Health Monitor - Main Application
Lightweight monitoring dashboard for Camunda BPM Platform clusters

Copyright (c) 2025 Champa Intelligence (https://champa-bpmn.com)
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from dotenv import load_dotenv
import urllib3

# Load environment variables
load_dotenv()

# Disable SSL warnings only if SSL verification is disabled
if os.getenv('SSL_VERIFY', 'false').lower() != 'true':
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Try to import JSON logger
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGING_AVAILABLE = True
except ImportError:
    jsonlogger = None
    JSON_LOGGING_AVAILABLE = False


def create_app(config_name=None):
    """
    Application factory pattern
    Creates and configures the Flask application

    Args:
        config_name: Configuration name (development, production, testing)

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)

    # Load configuration
    from config import get_config
    config = get_config(config_name)
    app.config.from_object(config)

    # Configure logging
    configure_logging(app, config)

    # Log startup
    logger = logging.getLogger('champa_monitor')
    logger.info("Camunda Health Monitor Starting")
    logger.info("=" * 60)

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    # Log configuration
    camunda_nodes = config.load_camunda_nodes()
    jmx_endpoints = config.load_jmx_endpoints()

    logger.info(f"Nodes: {len(camunda_nodes)}, JMX endpoints: {len(jmx_endpoints)}, JVM source: {config.JVM_METRICS_SOURCE}")
    logger.info(f"Database: {config.DB_CONFIG['user']}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['dbname']}")
    logger.info("=" * 60)

    # Initialize extensions and helpers
    initialize_extensions(app)

    # Register blueprints (routes)
    register_blueprints(app)

    # Register Jinja filters
    register_filters(app)

    logger.info("Application initialization complete")

    return app


def configure_logging(app, config):
    """Configure application logging"""
    # Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Determine logging format
    use_json = config.JSON_LOGGING and JSON_LOGGING_AVAILABLE

    if use_json:
        log_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        log_msg = "Using structured JSON logging"
    else:
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        log_msg = "Using standard text logging"

    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5
        )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Configure app logger
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)

    # Create general logger
    logger = logging.getLogger('champa_monitor')
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    logger.info(log_msg)


def initialize_extensions(app):
    """Initialize Flask extensions and application helpers"""
    from helpers.db_helper import init_db_pool
    from helpers.swagger_config import setup_swagger_ui
    from helpers.health_checks import create_kubernetes_probes, get_health_registry
    from helpers.shutdown_handler import get_shutdown_handler, register_shutdown_callback
    from config import Config

    logger = logging.getLogger('champa_monitor')

    # Initialize database
    try:
        init_db_pool(Config.DB_CONFIG, min_conn=1, max_conn=20)
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning("Application will start without database support")

    # Setup Swagger UI documentation
    setup_swagger_ui(app)

    # Create Kubernetes health probes
    try:
        from helpers.db_helper import get_db_helper
        db_helper_instance = get_db_helper()
    except RuntimeError:
        db_helper_instance = None

    create_kubernetes_probes(app, db_helper_instance, Config.load_camunda_nodes())

    # Register signal handlers for graceful shutdown
    shutdown_handler = get_shutdown_handler()
    shutdown_handler.register_signal_handlers()

    # Register database cleanup callback
    def cleanup_database():
        """Cleanup database connections"""
        try:
            from helpers.db_helper import get_db_helper
            db_helper = get_db_helper()
            logger.info("Closing database connection pool...")
            db_helper.close()
        except RuntimeError:
            pass  # Database helper was never initialized
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")

    register_shutdown_callback(cleanup_database, "Database cleanup")

    # Mark startup as complete
    health_registry = get_health_registry()
    health_registry.mark_startup_complete()


def register_blueprints(app):
    """Register Flask blueprints (route modules)"""
    from routes import register_blueprints as reg_bp
    reg_bp(app)


def register_filters(app):
    """Register Jinja template filters"""

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


# Create application instance
# This is what Gunicorn will import: gunicorn app:app
app = create_app()


# ============================================================
# Main Entry Point (for direct execution only)
# ============================================================

if __name__ == '__main__':
    logger = logging.getLogger('champa_monitor')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'

    logger.info("=" * 60)
    logger.info("Available endpoints:")
    logger.info(f"  Dashboard:    http://localhost:{port}/")
    logger.info(f"  API Health:   http://localhost:{port}/api/health")
    logger.info(f"  API Docs:     http://localhost:{port}/api/docs")
    logger.info(f"  Metrics:      http://localhost:{port}/metrics")
    logger.info(f"  Health Check: http://localhost:{port}/health")
    logger.info(f"  Readiness:    http://localhost:{port}/health/ready")
    logger.info("=" * 60)

    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        # Trigger graceful shutdown
        from helpers.health_checks import get_health_registry
        from helpers.shutdown_handler import get_shutdown_handler

        health_registry = get_health_registry()
        health_registry.mark_shutdown_initiated()

        shutdown_handler = get_shutdown_handler()
        shutdown_handler.initiate_shutdown()

