"""
Gunicorn configuration file for Camunda Health Monitor

Copyright (c) 2024-2025 Champa Intelligence (https://champa-bpmn.com)
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker processes
# For I/O-bound applications like this, use (2 x CPU cores) + 1
max_workers = multiprocessing.cpu_count() * 2 + 1
workers = int(os.getenv('GUNICORN_WORKERS', max_workers if max_workers < 8 else 8))
worker_class = 'sync'
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120  # Longer timeout for health collection operations
keepalive = 5

# Logging
accesslog = os.getenv('GUNICORN_ACCESS_LOG', 'logs/gunicorn-access.log')
errorlog = os.getenv('GUNICORN_ERROR_LOG', 'logs/gunicorn-error.log')
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'camunda-health-monitor'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed in future)
# keyfile = '/path/to/key.pem'
# certfile = '/path/to/cert.pem'

# Preload app for better performance and memory usage
# Note: With preload_app=True, the app is loaded in the master process
# Database connections are created per-worker in post_fork
preload_app = True

# Graceful timeout for shutdown
graceful_timeout = 30


# Server hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    print("=" * 60)
    print("Starting Camunda Health Monitor with Gunicorn...")
    print(f"Workers: {workers}")
    print(f"Bind: {bind}")
    print(f"Preload: {preload_app}")
    print("=" * 60)


def when_ready(server):
    """Called just after the server is started."""
    print(f"Camunda Health Monitor ready on {bind}")
    print(f"Workers: {workers}")
    print(f"Timeout: {timeout}s")
    print(f"Graceful timeout: {graceful_timeout}s")
    print("Application initialized and ready to serve requests")


def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    print("Reloading Camunda Health Monitor...")


def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    print(f"Worker {worker.pid} received INT or QUIT signal")


def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    print(f"Worker {worker.pid} received SIGABRT signal")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    # Close database connections in master process before forking
    # This prevents connection sharing between master and workers
    try:
        from helpers.db_helper import get_db_helper
        db_helper = get_db_helper()
        # Don't close here as we're using connection pooling
        # Each worker will get its own connections from the pool
        print(f"Preparing to fork worker {worker.pid}")
    except:
        pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"Worker {worker.pid} started")

    # Each worker marks itself as ready
    try:
        from helpers.health_checks import get_health_registry
        registry = get_health_registry()
        registry.mark_startup_complete()
        print(f"Worker {worker.pid} marked as ready")
    except Exception as e:
        print(f"Worker {worker.pid} initialization warning: {e}")


def pre_exec(server):
    """Called just before a new master process is forked."""
    print("Forking new master process...")


def on_exit(server):
    """Called just before exiting."""
    print("Shutting down Camunda Health Monitor...")
    from helpers.health_checks import get_health_registry
    from helpers.shutdown_handler import get_shutdown_handler

    try:
        # Mark shutdown initiated
        registry = get_health_registry()
        registry.mark_shutdown_initiated()

        # Cleanup resources
        shutdown_handler = get_shutdown_handler()
        for callback, name in shutdown_handler.shutdown_callbacks:
            try:
                print(f"Executing shutdown callback: {name}")
                callback()
            except Exception as e:
                print(f"Error in shutdown callback '{name}': {e}")
    except:
        pass
