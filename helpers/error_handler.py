"""
Error Handler Module
Centralized error handling, logging, and response formatting
"""
import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from functools import wraps
from flask import jsonify, request
from datetime import datetime, timezone

logger = logging.getLogger('champa_monitor.errors')


class CamundaMonitorError(Exception):
    """Base exception for Camunda Monitor errors"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DatabaseError(CamundaMonitorError):
    """Database-related errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=503, details=details)


class APIError(CamundaMonitorError):
    """External API call errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=502, details=details)


class ValidationError(CamundaMonitorError):
    """Input validation errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=400, details=details)


class ConfigurationError(CamundaMonitorError):
    """Configuration errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=500, details=details)


class ErrorHandler:
    """Centralized error handler for the application"""

    @staticmethod
    def format_error_response(error: Exception, status_code: int = 500,
                            include_traceback: bool = False) -> Tuple[Dict[str, Any], int]:
        """
        Format an error into a standardized JSON response

        Args:
            error: The exception to format
            status_code: HTTP status code
            include_traceback: Whether to include stack trace in response

        Returns:
            Tuple of (response_dict, status_code)
        """
        response = {
            "error": True,
            "message": str(error),
            "type": type(error).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": request.path if request else None
        }

        # Add details if it's our custom exception
        if isinstance(error, CamundaMonitorError):
            if error.details:
                response["details"] = error.details
            status_code = error.status_code

        # Include traceback in development mode
        if include_traceback:
            response["traceback"] = traceback.format_exc()

        return response, status_code

    @staticmethod
    def log_error(error: Exception, context: Optional[str] = None,
                 level: int = logging.ERROR, include_traceback: bool = True):
        """
        Log an error with appropriate context

        Args:
            error: The exception to log
            context: Additional context string
            level: Logging level
            include_traceback: Whether to include stack trace
        """
        message = f"{context}: {error}" if context else str(error)

        extra_data = {
            "error_type": type(error).__name__,
            "path": request.path if request else None,
            "method": request.method if request else None
        }

        if isinstance(error, CamundaMonitorError) and error.details:
            extra_data["details"] = error.details

        if include_traceback:
            logger.log(level, message, exc_info=True, extra=extra_data)
        else:
            logger.log(level, message, extra=extra_data)

    @staticmethod
    def handle_exception(error: Exception, context: Optional[str] = None,
                        include_traceback: bool = False) -> Tuple[Dict[str, Any], int]:
        """
        Handle an exception: log it and format response

        Args:
            error: The exception to handle
            context: Additional context for logging
            include_traceback: Whether to include stack trace in response

        Returns:
            Tuple of (response_dict, status_code)
        """
        # Log the error
        ErrorHandler.log_error(error, context)

        # Format and return response
        return ErrorHandler.format_error_response(error, include_traceback=include_traceback)


def handle_errors(context: Optional[str] = None, include_traceback: bool = False):
    """
    Decorator to handle errors in route handlers

    Args:
        context: Context string for error logging
        include_traceback: Whether to include stack trace in response

    Usage:
        @app.route('/api/data')
        @handle_errors(context='Fetching data')
        def get_data():
            return jsonify(data)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CamundaMonitorError as e:
                response, status_code = ErrorHandler.handle_exception(
                    e, context or func.__name__, include_traceback
                )
                return jsonify(response), status_code
            except Exception as e:
                response, status_code = ErrorHandler.handle_exception(
                    e, context or func.__name__, include_traceback
                )
                return jsonify(response), status_code
        return wrapper
    return decorator


def safe_execute(func, default_value=None, log_errors=True, context: Optional[str] = None):
    """
    Safely execute a function and return default value on error

    Args:
        func: Function to execute
        default_value: Value to return on error
        log_errors: Whether to log errors
        context: Context for error logging

    Returns:
        Function result or default_value on error

    Usage:
        result = safe_execute(lambda: risky_operation(), default_value=[], context='Risky op')
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            ErrorHandler.log_error(e, context or 'Safe execute', level=logging.WARNING)
        return default_value


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance

    States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
    """

    CLOSED = 'CLOSED'
    OPEN = 'OPEN'
    HALF_OPEN = 'HALF_OPEN'

    def __init__(self, failure_threshold: int = 5, timeout: int = 60, name: str = "default"):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds before attempting to close circuit
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time = None
        self.state = self.CLOSED

    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == self.OPEN:
            if self._should_attempt_reset():
                self.state = self.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}': Attempting reset (HALF_OPEN)")
            else:
                raise CamundaMonitorError(
                    f"Circuit breaker '{self.name}' is OPEN",
                    status_code=503,
                    details={"circuit_breaker": self.name, "state": self.state}
                )

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        import time
        return (self.last_failure_time and
                time.time() - self.last_failure_time >= self.timeout)

    def on_success(self):
        """Handle successful call"""
        if self.state == self.HALF_OPEN:
            logger.info(f"Circuit breaker '{self.name}': Recovered (CLOSED)")
        self.failure_count = 0
        self.state = self.CLOSED

    def on_failure(self):
        """Handle failed call"""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            if self.state != self.OPEN:
                logger.warning(
                    f"Circuit breaker '{self.name}': Opened after {self.failure_count} failures"
                )
            self.state = self.OPEN

    def reset(self):
        """Manually reset the circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = self.CLOSED
        logger.info(f"Circuit breaker '{self.name}': Manually reset")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "timeout": self.timeout
        }

