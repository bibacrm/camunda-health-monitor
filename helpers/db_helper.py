"""
Database Helper Module
Handles all database connection pooling, query execution, and data access operations
"""
import logging
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger('champa_monitor.db')


class DatabaseHelper:
    """Database connection pool manager with health checks and error handling"""

    def __init__(self, db_config: Dict[str, Any], min_conn: int = 1, max_conn: int = 20):
        """
        Initialize database connection pool

        Args:
            db_config: Database configuration dictionary
            min_conn: Minimum number of connections
            max_conn: Maximum number of connections
        """
        self.db_config = db_config
        self.min_conn = min_conn
        self.max_conn = max_conn
        self._pool: Optional[pool.SimpleConnectionPool] = None
        self._initialized = False

    def initialize(self):
        """Initialize the database connection pool"""
        import threading

        if self._initialized:
            logger.debug("Database pool already initialized")
            return

        init_timeout = 10  # 10 seconds timeout
        pool_created = []
        error_occurred = []

        def create_pool():
            """Create pool in separate thread"""
            try:
                p = pool.SimpleConnectionPool(
                    self.min_conn,
                    self.max_conn,
                    **self.db_config
                )
                pool_created.append(p)
            except Exception as e:
                error_occurred.append(e)

        try:
            logger.info(f"Connecting to database {self.db_config.get('user')}@{self.db_config.get('host')}:{self.db_config.get('port')}/{self.db_config.get('dbname')}...")

            # Create pool in a thread with timeout
            thread = threading.Thread(target=create_pool, daemon=True)
            thread.start()
            thread.join(timeout=init_timeout)

            if thread.is_alive():
                logger.error(f"Database connection timeout ({init_timeout}s) - PostgreSQL not responding")
                raise TimeoutError(f"Database connection timed out after {init_timeout} seconds")

            if error_occurred:
                raise error_occurred[0]

            if not pool_created:
                raise RuntimeError("Pool creation failed")

            self._pool = pool_created[0]

            # CRITICAL: Set _initialized = True BEFORE testing the connection
            self._initialized = True

            # Test the pool
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()

            logger.info(f"Database connected ({self.min_conn}-{self.max_conn} connection pool)")

        except TimeoutError:
            raise
        except Exception as e:
            self._initialized = False
            logger.error(f"Database connection failed: {e}")
            raise

    def close(self):
        """Close all connections in the pool"""
        if self._pool:
            try:
                self._pool.closeall()
                logger.info("Database connection pool closed")
                self._initialized = False
            except Exception as e:
                logger.error(f"Error closing database pool: {e}", exc_info=True)

    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool with automatic cleanup

        Yields:
            Database connection

        Usage:
            with db_helper.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        if not self._initialized:
            self.initialize()

        conn = None
        try:
            conn = self._pool.getconn()

            # Validate connection is still alive
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            except Exception as e:
                # Connection is bad, close it and get a new one
                logger.debug(f"Stale connection detected, replacing: {e}")
                try:
                    conn.close()
                except:
                    pass
                conn = self._pool.getconn()

            yield conn
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {e}", exc_info=True)
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
                # Mark connection as bad - close instead of returning to pool
                try:
                    conn.close()
                except:
                    pass
                conn = None
            raise
        finally:
            if conn:
                try:
                    self._pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as list of dictionaries

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            List of dictionaries with query results
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)

                    if cursor.description:
                        results = cursor.fetchall()
                        return [dict(row) for row in results] if results else []
                    return []
        except Exception as e:
            logger.error(f"Database query error: {e}", exc_info=True)
            logger.debug(f"Failed query: {query}")
            return []

    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            Number of affected rows
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)

                    affected_rows = cursor.rowcount
                    conn.commit()
                    return affected_rows
        except Exception as e:
            logger.error(f"Database update error: {e}", exc_info=True)
            logger.debug(f"Failed query: {query}")
            raise

    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """
        Execute a query with multiple parameter sets

        Args:
            query: SQL query string
            params_list: List of parameter tuples

        Returns:
            Number of affected rows
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.executemany(query, params_list)
                    affected_rows = cursor.rowcount
                    conn.commit()
                    return affected_rows
        except Exception as e:
            logger.error(f"Database executemany error: {e}", exc_info=True)
            raise

    def test_connection(self) -> Tuple[bool, Optional[float]]:
        """
        Test database connection and measure latency

        Returns:
            Tuple of (success: bool, latency_ms: float)
        """
        import time
        try:
            start_time = time.time()
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            latency_ms = (time.time() - start_time) * 1000
            return True, latency_ms
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False, None

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics

        Returns:
            Dictionary with pool statistics
        """
        if not self._pool:
            return {
                "initialized": False,
                "min_connections": self.min_conn,
                "max_connections": self.max_conn
            }

        # Note: psycopg2.pool doesn't expose detailed stats directly
        # We can only provide basic info
        return {
            "initialized": self._initialized,
            "min_connections": self.min_conn,
            "max_connections": self.max_conn,
            "status": "healthy" if self._initialized else "uninitialized"
        }


# Convenience functions for backward compatibility
_default_db_helper: Optional[DatabaseHelper] = None


def init_db_pool(db_config: Dict[str, Any], min_conn: int = 1, max_conn: int = 20) -> DatabaseHelper:
    """Initialize and return the default database helper instance"""
    global _default_db_helper

    # If already initialized, return existing instance
    if _default_db_helper is not None:
        logger.debug("Database pool already initialized, returning existing instance")
        return _default_db_helper

    _default_db_helper = DatabaseHelper(db_config, min_conn, max_conn)
    _default_db_helper.initialize()
    return _default_db_helper


def get_db_helper() -> DatabaseHelper:
    """Get the default database helper instance"""
    if _default_db_helper is None:
        raise RuntimeError("Database helper not initialized. Call init_db_pool() first.")
    return _default_db_helper


def execute_query(query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    """Execute a query using the default database helper"""
    return get_db_helper().execute_query(query, params)


def execute_update(query: str, params: Optional[Tuple] = None) -> int:
    """Execute an update using the default database helper"""
    return get_db_helper().execute_update(query, params)

