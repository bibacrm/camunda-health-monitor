"""
Gunicorn configuration file for Camunda Health Monitor

Copyright (c) 2025 Champa Intelligence (https://champa-bpmn.com)
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'sync'
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
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

# SSL (if needed)
# keyfile = '/path/to/key.pem'
# certfile = '/path/to/cert.pem'

# Preload app for better performance
preload_app = True

# Server hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    print("Starting Camunda Health Monitor...")

def on_reload(server):
    """Called when the server is being reloaded."""
    print("Reloading Camunda Health Monitor...")

def when_ready(server):
    """Called just after the server is started."""
    print(f"Camunda Health Monitor ready on {bind}")
    print(f"Workers: {workers}")
    print(f"Timeout: {timeout}s")

def on_exit(server):
    """Called just before exiting."""
    print("Shutting down Camunda Health Monitor...")
