"""
OpenAPI/Swagger Configuration
Generates interactive API documentation
"""
from flask import jsonify
import os
import logging

logger = logging.getLogger('champa_monitor.swagger')


def get_openapi_spec(app_version: str = "1.0.0"):
    """
    Generate OpenAPI 3.0 specification for the API

    Args:
        app_version: Application version

    Returns:
        OpenAPI specification dictionary
    """
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Camunda Health Monitor API",
            "description": """
# Camunda Health Monitor API

A comprehensive monitoring API for Camunda BPM Platform clusters.

## Features

- **Real-time Monitoring**: Track process instances, tasks, incidents, and jobs
- **Cluster Health**: Monitor multiple Camunda nodes with JVM metrics
- **Database Metrics**: Track database performance and storage
- **Prometheus Export**: Export metrics in Prometheus format
- **Health Probes**: Kubernetes-compatible health checks

## Authentication

Currently, the API does not require authentication. For production use, please implement 
appropriate authentication and authorization mechanisms.

## Rate Limiting

No rate limiting is currently enforced. This should be implemented for production deployments.
            """,
            "version": app_version,
            "contact": {
                "name": "Champa Intelligence",
                "url": "https://champa-bpmn.com"
            }
        },
        "servers": [
            {
                "url": "/",
                "description": "Current server"
            }
        ],
        "tags": [
            {
                "name": "health",
                "description": "Health check and monitoring endpoints"
            },
            {
                "name": "metrics",
                "description": "Camunda and database metrics"
            },
            {
                "name": "kubernetes",
                "description": "Kubernetes health probes"
            },
            {
                "name": "prometheus",
                "description": "Prometheus metrics export"
            }
        ],
        "paths": {
            "/api/health": {
                "get": {
                    "tags": ["health"],
                    "summary": "Get comprehensive cluster health status",
                    "description": "Returns detailed health information about all Camunda nodes, including metrics, JVM status, and workload distribution.",
                    "operationId": "getHealth",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthResponse"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "tags": ["health"],
                    "summary": "Basic health check",
                    "description": "Simple health check endpoint returning OK status.",
                    "operationId": "basicHealth",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {
                                                "type": "string",
                                                "example": "OK"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health/live": {
                "get": {
                    "tags": ["kubernetes"],
                    "summary": "Liveness probe",
                    "description": "Kubernetes liveness probe - checks if the application is running.",
                    "operationId": "livenessProbe",
                    "responses": {
                        "200": {
                            "description": "Application is alive",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ProbeResponse"
                                    }
                                }
                            }
                        },
                        "503": {
                            "description": "Application is shutting down",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ProbeResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health/ready": {
                "get": {
                    "tags": ["kubernetes"],
                    "summary": "Readiness probe",
                    "description": "Kubernetes readiness probe - checks if the application is ready to serve traffic.",
                    "operationId": "readinessProbe",
                    "responses": {
                        "200": {
                            "description": "Application is ready",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ProbeResponse"
                                    }
                                }
                            }
                        },
                        "503": {
                            "description": "Application is not ready",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ProbeResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health/startup": {
                "get": {
                    "tags": ["kubernetes"],
                    "summary": "Startup probe",
                    "description": "Kubernetes startup probe - checks if the application has started successfully.",
                    "operationId": "startupProbe",
                    "responses": {
                        "200": {
                            "description": "Application has started",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ProbeResponse"
                                    }
                                }
                            }
                        },
                        "503": {
                            "description": "Application is still starting",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ProbeResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health/detailed": {
                "get": {
                    "tags": ["kubernetes"],
                    "summary": "Detailed health check",
                    "description": "Comprehensive health check with all registered checks and their individual results.",
                    "operationId": "detailedHealth",
                    "responses": {
                        "200": {
                            "description": "System is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DetailedHealthResponse"
                                    }
                                }
                            }
                        },
                        "503": {
                            "description": "System is unhealthy or degraded",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DetailedHealthResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/metrics/stuck-instances": {
                "get": {
                    "tags": ["metrics"],
                    "summary": "Get stuck process instances",
                    "description": "Returns process instances that have been running longer than the configured threshold.",
                    "operationId": "getStuckInstances",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "stuck_instances": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object"
                                                }
                                            },
                                            "timestamp": {
                                                "type": "string",
                                                "format": "date-time"
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/metrics/pending-messages": {
                "get": {
                    "tags": ["metrics"],
                    "summary": "Get pending message subscriptions",
                    "description": "Returns message event subscriptions waiting for correlation.",
                    "operationId": "getPendingMessages",
                    "responses": {
                        "200": {
                            "description": "Successful response"
                        },
                        "500": {
                            "description": "Error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/metrics/pending-signals": {
                "get": {
                    "tags": ["metrics"],
                    "summary": "Get pending signal subscriptions",
                    "description": "Returns signal event subscriptions waiting for correlation.",
                    "operationId": "getPendingSignals",
                    "responses": {
                        "200": {
                            "description": "Successful response"
                        },
                        "500": {
                            "description": "Error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/metrics/job-throughput": {
                "get": {
                    "tags": ["metrics"],
                    "summary": "Get job execution throughput",
                    "description": "Returns the number of jobs executed per minute over the last 10 minutes.",
                    "operationId": "getJobThroughput",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "jobs_executed_per_min": {
                                                "type": "number",
                                                "example": 42.5
                                            },
                                            "timestamp": {
                                                "type": "string",
                                                "format": "date-time"
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/metrics/database": {
                "get": {
                    "tags": ["metrics"],
                    "summary": "Get database metrics",
                    "description": "Returns database storage and performance metrics.",
                    "operationId": "getDatabaseMetrics",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DatabaseMetrics"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/metrics": {
                "get": {
                    "tags": ["prometheus"],
                    "summary": "Prometheus metrics export",
                    "description": "Export all metrics in Prometheus text format for scraping.",
                    "operationId": "prometheusMetrics",
                    "responses": {
                        "200": {
                            "description": "Metrics in Prometheus format",
                            "content": {
                                "text/plain": {
                                    "schema": {
                                        "type": "string"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Error",
                            "content": {
                                "text/plain": {
                                    "schema": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "boolean",
                            "example": True
                        },
                        "message": {
                            "type": "string",
                            "example": "An error occurred"
                        },
                        "type": {
                            "type": "string",
                            "example": "Exception"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "path": {
                            "type": "string",
                            "example": "/api/health"
                        }
                    }
                },
                "ProbeResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["alive", "ready", "started", "starting", "shutdown", "not_ready"]
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "DetailedHealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "degraded", "unhealthy"]
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "checks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string"
                                    },
                                    "status": {
                                        "type": "string"
                                    },
                                    "critical": {
                                        "type": "boolean"
                                    },
                                    "timestamp": {
                                        "type": "string",
                                        "format": "date-time"
                                    },
                                    "details": {
                                        "type": "object"
                                    }
                                }
                            }
                        },
                        "summary": {
                            "type": "object",
                            "properties": {
                                "total": {
                                    "type": "integer"
                                },
                                "healthy": {
                                    "type": "integer"
                                },
                                "critical_failures": {
                                    "type": "integer"
                                },
                                "non_critical_failures": {
                                    "type": "integer"
                                }
                            }
                        }
                    }
                },
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "example": "RUNNING"
                        },
                        "cluster_status": {
                            "type": "object"
                        },
                        "cluster_nodes": {
                            "type": "array",
                            "items": {
                                "type": "object"
                            }
                        },
                        "totals": {
                            "type": "object"
                        },
                        "db_metrics": {
                            "type": "object"
                        }
                    }
                },
                "DatabaseMetrics": {
                    "type": "object",
                    "properties": {
                        "table_sizes": {
                            "type": "array",
                            "items": {
                                "type": "object"
                            }
                        },
                        "total_size_mb": {
                            "type": "number"
                        },
                        "index_usage": {
                            "type": "array",
                            "items": {
                                "type": "object"
                            }
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                }
            }
        }
    }


def setup_swagger_ui(app):
    """
    Setup Swagger UI for interactive API documentation

    Args:
        app: Flask application instance
    """

    @app.route('/api/docs')
    def swagger_ui():
        """Serve Swagger UI"""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Camunda Health Monitor API - Swagger UI</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.10.0/swagger-ui.css">
    <style>
        body { margin: 0; padding: 0; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.10.0/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.10.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: '/api/openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                layout: "StandaloneLayout"
            });
        };
    </script>
</body>
</html>
        """
        return html

    @app.route('/api/openapi.json')
    def openapi_spec():
        """Serve OpenAPI specification"""
        version = os.getenv('APP_VERSION', '1.0.0')
        spec = get_openapi_spec(version)
        return jsonify(spec)

    logger.debug("Swagger UI configured at /api/docs")


