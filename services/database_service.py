"""
Database Service
Handles all database-related operations and metrics collection
"""
import logging
from flask import current_app
from helpers.error_handler import safe_execute
from helpers.db_helper import execute_query

logger = logging.getLogger('champa_monitor.database_service')


def timed_cache(seconds=60):
    """Cache decorator with time-based expiration"""
    from functools import wraps
    from datetime import datetime, timedelta

    def decorator(func):
        cache = {}
        cache_time = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)

            if key in cache and key in cache_time:
                if datetime.now() - cache_time[key] < timedelta(seconds=seconds):
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[key]

            logger.debug(f"Cache miss for {func.__name__}, executing...")
            result = func(*args, **kwargs)
            cache[key] = result
            cache_time[key] = datetime.now()

            if len(cache) > 100:
                oldest_key = min(cache_time.keys(), key=lambda k: cache_time[k])
                del cache[oldest_key]
                del cache_time[oldest_key]

            return result

        wrapper.cache_clear = lambda: cache.clear() or cache_time.clear()
        return wrapper

    return decorator


@timed_cache(seconds=120)
def collect_database_metrics():
    """Collect database performance and storage metrics"""
    # Table sizes - use safe_execute for error tolerance
    table_sizes = safe_execute(
        lambda: execute_query("""
            SELECT relname AS table_name, pg_total_relation_size(relid) AS size_bytes
            FROM pg_catalog.pg_statio_user_tables
            WHERE relname LIKE 'act_h%' OR relname LIKE 'act_r%'
            ORDER BY size_bytes DESC
            LIMIT 10
        """),
        default_value=[],
        context="Fetching table sizes"
    )

    # Archivable instances
    archive_days = current_app.config.get('DB_ARCHIVE_THRESHOLD_DAYS', 90)
    archivable = safe_execute(
        lambda: execute_query(f"""
            SELECT count(*) AS count
            FROM act_hi_procinst
            WHERE end_time_ IS NOT NULL
            AND end_time_ < now() - interval '{archive_days} days'
        """),
        default_value=[],
        context="Fetching archivable instances"
    )

    # Slow queries (requires pg_stat_statements extension)
    slow_queries = safe_execute(
        lambda: execute_query("""
            SELECT calls, 
                   ROUND(mean_exec_time::numeric, 2) AS avg_ms, 
                   ROUND(max_exec_time::numeric, 2) AS max_ms,
                   LEFT(query, 120) AS query_preview
            FROM pg_stat_statements
            WHERE query LIKE '%act_%'
            ORDER BY mean_exec_time DESC
            LIMIT 10
        """),
        default_value=[],
        log_errors=False,  # pg_stat_statements is optional
        context="Fetching slow queries"
    )

    return {
        "table_sizes": table_sizes,
        "archivable_instances": archivable[0]['count'] if archivable else 0,
        "slow_queries": slow_queries
    }

