"""
Graceful Shutdown Handler
Handles application shutdown gracefully, ensuring connections are closed properly
"""
import logging
import signal
import sys
from typing import Callable, List

logger = logging.getLogger('champa_monitor.shutdown')


class ShutdownHandler:
    """Manages graceful shutdown of the application"""

    def __init__(self):
        self.shutdown_callbacks: List[tuple] = []
        self.shutdown_initiated = False

    def register_shutdown_callback(self, callback: Callable, name: str = None):
        """
        Register a callback to be executed during shutdown

        Args:
            callback: Function to call during shutdown
            name: Optional name for logging
        """
        self.shutdown_callbacks.append((callback, name or callback.__name__))
        logger.debug(f"Registered shutdown callback: {name or callback.__name__}")

    def initiate_shutdown(self, signum=None, frame=None):
        """
        Initiate graceful shutdown

        Args:
            signum: Signal number (if called from signal handler)
            frame: Current stack frame (if called from signal handler)
        """
        if self.shutdown_initiated:
            logger.warning("Shutdown already initiated, forcing exit...")
            sys.exit(1)

        self.shutdown_initiated = True

        if signum:
            signal_name = signal.Signals(signum).name
            logger.info(f"Received signal {signal_name} ({signum}), initiating graceful shutdown...")
        else:
            logger.info("Initiating graceful shutdown...")

        # Execute all shutdown callbacks
        for callback, name in self.shutdown_callbacks:
            try:
                logger.info(f"Executing shutdown callback: {name}")
                callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback '{name}': {e}", exc_info=True)

        logger.info("Graceful shutdown complete")
        sys.exit(0)

    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        # Handle SIGTERM (Docker/K8s sends this)
        signal.signal(signal.SIGTERM, self.initiate_shutdown)

        # Handle SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.initiate_shutdown)

        logger.debug("Signal handlers registered (SIGTERM, SIGINT)")


# Global shutdown handler instance
_shutdown_handler = ShutdownHandler()


def get_shutdown_handler() -> ShutdownHandler:
    """Get the global shutdown handler instance"""
    return _shutdown_handler


def register_shutdown_callback(callback: Callable, name: str = None):
    """Register a callback to be executed during shutdown"""
    _shutdown_handler.register_shutdown_callback(callback, name)


def register_signal_handlers():
    """Register signal handlers for graceful shutdown"""
    _shutdown_handler.register_signal_handlers()

