"""
Health Check Module
Kubernetes-compatible health, readiness, and liveness probes
"""
import logging
from typing import Dict, Any, List, Callable
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger('champa_monitor.health')


class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Individual health check"""

    def __init__(self, name: str, check_func: Callable, critical: bool = True):
        """
        Initialize health check

        Args:
            name: Check name
            check_func: Function that returns (status: bool, details: dict)
            critical: Whether this check is critical for overall health
        """
        self.name = name
        self.check_func = check_func
        self.critical = critical
        self.last_check_time = None
        self.last_status = None
        self.last_details = None

    def execute(self) -> Dict[str, Any]:
        """
        Execute the health check

        Returns:
            Dictionary with check results
        """
        try:
            status, details = self.check_func()
            self.last_check_time = datetime.now(timezone.utc)
            self.last_status = status
            self.last_details = details

            return {
                "name": self.name,
                "status": "healthy" if status else "unhealthy",
                "critical": self.critical,
                "timestamp": self.last_check_time.isoformat(),
                "details": details or {}
            }
        except Exception as e:
            logger.error(f"Health check '{self.name}' failed: {e}", exc_info=True)
            self.last_check_time = datetime.now(timezone.utc)
            self.last_status = False
            self.last_details = {"error": str(e)}

            return {
                "name": self.name,
                "status": "unhealthy",
                "critical": self.critical,
                "timestamp": self.last_check_time.isoformat(),
                "details": {"error": str(e)}
            }


class HealthCheckRegistry:
    """Registry for managing multiple health checks"""

    def __init__(self):
        self.checks: List[HealthCheck] = []
        self.startup_complete = False
        self.shutdown_initiated = False

    def register(self, name: str, check_func: Callable, critical: bool = True):
        """
        Register a new health check

        Args:
            name: Check name
            check_func: Function that returns (status: bool, details: dict)
            critical: Whether this check is critical
        """
        check = HealthCheck(name, check_func, critical)
        self.checks.append(check)
        logger.debug(f"Registered health check: {name} (critical={critical})")

    def execute_all(self) -> Dict[str, Any]:
        """
        Execute all registered health checks

        Returns:
            Dictionary with overall health status and individual check results
        """
        results = []
        critical_failures = []
        non_critical_failures = []

        for check in self.checks:
            result = check.execute()
            results.append(result)

            if result["status"] != "healthy":
                if check.critical:
                    critical_failures.append(check.name)
                else:
                    non_critical_failures.append(check.name)

        # Determine overall status
        if critical_failures:
            overall_status = HealthStatus.UNHEALTHY.value
        elif non_critical_failures:
            overall_status = HealthStatus.DEGRADED.value
        else:
            overall_status = HealthStatus.HEALTHY.value

        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": results,
            "summary": {
                "total": len(self.checks),
                "healthy": len(self.checks) - len(critical_failures) - len(non_critical_failures),
                "critical_failures": len(critical_failures),
                "non_critical_failures": len(non_critical_failures)
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if system is healthy (all critical checks pass)

        Returns:
            True if healthy, False otherwise
        """
        for check in self.checks:
            if check.critical and check.last_status is False:
                return False
        return True

    def mark_startup_complete(self):
        """Mark that application startup is complete"""
        self.startup_complete = True
        logger.debug("Application startup marked as complete")

    def mark_shutdown_initiated(self):
        """Mark that application shutdown has been initiated"""
        self.shutdown_initiated = True
        logger.info("Application shutdown marked as initiated")


# Global registry instance
_health_registry = HealthCheckRegistry()


def get_health_registry() -> HealthCheckRegistry:
    """Get the global health check registry"""
    return _health_registry


def create_kubernetes_probes(app, db_helper, camunda_nodes: Dict[str, str]):
    """
    Create Kubernetes-compatible health check endpoints

    Args:
        app: Flask application
        db_helper: Database helper instance
        camunda_nodes: Dictionary of Camunda node configurations
    """
    registry = get_health_registry()

    # Register health checks
    def check_database():
        """Check database connectivity"""
        if db_helper is None:
            return False, {"error": "Database helper not initialized"}
        try:
            success, latency = db_helper.test_connection()
            return success, {"latency_ms": latency}
        except:
            return False, {"error": "Database unreachable"}

    def check_camunda_nodes():
        """Check if at least one Camunda node is reachable"""
        import requests
        from helpers.error_handler import safe_execute
        from config import Config

        # Get auth and SSL settings from config
        camunda_auth = Config.get_camunda_auth()
        ssl_verify = Config.SSL_VERIFY

        reachable_nodes = 0
        for name, url in camunda_nodes.items():
            def check_node():
                response = requests.get(
                    f"{url}/engine",
                    auth=camunda_auth,
                    timeout=5,
                    verify=ssl_verify
                )
                return response.status_code == 200

            if safe_execute(check_node, default_value=False, log_errors=False):
                reachable_nodes += 1

        total_nodes = len(camunda_nodes)
        return reachable_nodes > 0, {
            "reachable_nodes": reachable_nodes,
            "total_nodes": total_nodes
        }

    def check_memory():
        """Check system memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            # Consider unhealthy if > 95% memory used
            return memory.percent < 95, {
                "percent_used": memory.percent,
                "available_mb": memory.available / (1024 * 1024)
            }
        except ImportError:
            # psutil not available, skip check
            return True, {"message": "psutil not available"}
        except:
            return False, {"error": "Failed to check memory"}

    # Register checks
    registry.register("database", check_database, critical=True)
    registry.register("camunda_nodes", check_camunda_nodes, critical=True)
    registry.register("memory", check_memory, critical=False)

    # Liveness probe - Is the application running?
    @app.route('/health/live')
    def liveness_probe():
        """
        Kubernetes liveness probe
        Returns 200 if application is running, 503 if shutdown initiated
        """
        if registry.shutdown_initiated:
            return {
                "status": "shutdown",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 503

        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, 200

    # Readiness probe - Is the application ready to serve traffic?
    @app.route('/health/ready')
    def readiness_probe():
        """
        Kubernetes readiness probe
        Returns 200 if ready to serve traffic, 503 otherwise
        """
        if not registry.startup_complete:
            return {
                "status": "starting",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 503

        if registry.shutdown_initiated:
            return {
                "status": "shutdown",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 503

        # Execute critical checks only for readiness
        critical_check_results = []
        all_critical_healthy = True

        for check in registry.checks:
            if check.critical:
                result = check.execute()
                critical_check_results.append(result)
                if result["status"] != "healthy":
                    all_critical_healthy = False

        if all_critical_healthy:
            return {
                "status": "ready",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": critical_check_results
            }, 200
        else:
            return {
                "status": "not_ready",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": critical_check_results
            }, 503

    # Startup probe - Has the application started successfully?
    @app.route('/health/startup')
    def startup_probe():
        """
        Kubernetes startup probe
        Returns 200 if application has started successfully
        """
        if registry.startup_complete:
            return {
                "status": "started",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 200
        else:
            return {
                "status": "starting",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 503

    # Detailed health check endpoint
    @app.route('/health/detailed')
    def detailed_health():
        """
        Detailed health check with all registered checks
        Returns overall status and individual check results
        """
        result = registry.execute_all()
        status_code = 200 if result["status"] == "healthy" else 503
        return result, status_code

    logger.debug("Kubernetes health probes configured")


