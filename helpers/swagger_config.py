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
                "name": "ai",
                "description": "AI/ML-powered analytics and insights"
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
                    "summary": "Prometheus metrics export (includes AI metrics)",
                    "description": """Export all metrics in Prometheus text format for scraping.
                    
**Includes:**
- Cluster metrics (nodes, instances, tasks, incidents)
- Per-node metrics (status, response time, JVM health)
- Database metrics (latency, connections, utilization)
- **AI/ML metrics** (health score, anomalies detected, node performance scores)

**Performance:** < 300ms total (100ms AI overhead). AI metrics use lightweight COUNT/AVG queries only.

**AI Metrics Exported:**
- `camunda_ai_health_score` - Overall cluster health (0-100)
- `camunda_ai_anomalies_detected` - Number of process anomalies
- `camunda_ai_anomalies_critical` - Critical anomalies requiring attention
- `camunda_ai_node_performance_score{node}` - Per-node performance score (0-100)
                    """,
                    "operationId": "prometheusMetrics",
                    "responses": {
                        "200": {
                            "description": "Metrics in Prometheus format",
                            "content": {
                                "text/plain": {
                                    "schema": {
                                        "type": "string"
                                    },
                                    "example": """# HELP camunda_active_instances Active process instances
# TYPE camunda_active_instances gauge
camunda_active_instances 1234

# HELP camunda_ai_health_score AI-calculated cluster health score (0-100)
# TYPE camunda_ai_health_score gauge
camunda_ai_health_score 87.5

# HELP camunda_ai_anomalies_detected Number of process anomalies detected
# TYPE camunda_ai_anomalies_detected gauge
camunda_ai_anomalies_detected 3"""
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
            },
            "/api/ai/health-score": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Get AI-powered cluster health score",
                    "description": "Returns an overall health score (0-100) for the Camunda cluster with contributing factors and grade.",
                    "operationId": "getAIHealthScore",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AIHealthScore"
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
            "/api/ai/anomalies": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Detect process execution anomalies",
                    "description": "Uses statistical analysis to detect process definitions with abnormal execution times based on historical data.",
                    "operationId": "detectAnomalies",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AIAnomalies"
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
            "/api/ai/incident-patterns": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Analyze incident patterns",
                    "description": "Clusters and analyzes incident patterns to identify common failure scenarios and their root causes.",
                    "operationId": "analyzeIncidentPatterns",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AIIncidentPatterns"
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
            "/api/ai/bottlenecks": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Identify process bottlenecks",
                    "description": "Identifies activities that significantly slow down process execution and calculates their impact.",
                    "operationId": "identifyBottlenecks",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AIBottlenecks"
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
            "/api/ai/job-predictions": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Predict job failure probabilities",
                    "description": "Predicts which job types are likely to fail based on historical execution patterns.",
                    "operationId": "predictJobFailures",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AIJobPredictions"
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
            "/api/ai/node-performance": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Analyze node performance rankings",
                    "description": "Ranks cluster nodes by performance score based on JVM health, response time, and workload metrics.",
                    "operationId": "analyzeNodePerformance",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AINodePerformance"
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
            "/api/ai/process-leaderboard": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Get process performance leaderboard",
                    "description": "Ranks process definitions by performance metrics including execution time, completion rate, and efficiency.",
                    "operationId": "getProcessLeaderboard",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AIProcessLeaderboard"
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
            "/api/ai/sla-predictions": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Predict SLA breaches",
                    "description": "Predicts user tasks that are likely to breach SLA based on their current wait time and historical patterns.",
                    "operationId": "predictSLABreaches",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AISLAPredictions"
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
            "/api/ai/insights": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Get comprehensive AI insights",
                    "description": "Returns all AI/ML analytics in a single call including health score, anomalies, bottlenecks, predictions, and actionable recommendations.",
                    "operationId": "getAIInsights",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AIInsights"
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
            "/api/ai/stuck-activities-smart": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Find stuck activities using smart detection",
                    "description": "Advanced stuck activity detection using statistical percentile thresholds (P95). Identifies activities taking abnormally long based on historical execution patterns rather than hardcoded timeouts.",
                    "operationId": "getStuckActivitiesSmart",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "stuck_activities": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "activity_instance_id": {"type": "string"},
                                                        "process_instance_id": {"type": "string"},
                                                        "process_key": {"type": "string"},
                                                        "activity_name": {"type": "string"},
                                                        "stuck_for_hours": {"type": "number"},
                                                        "duration_ratio": {"type": "number"},
                                                        "severity": {"type": "string", "enum": ["critical", "high", "medium"]},
                                                        "message": {"type": "string"}
                                                    }
                                                }
                                            },
                                            "total_found": {"type": "integer"},
                                            "threshold_percentile": {"type": "integer"},
                                            "timestamp": {"type": "string", "format": "date-time"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/ai/predict-duration/{process_def_key}": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Predict process duration using ML",
                    "description": "Predict how long a process will take to complete using machine learning. Based on historical execution patterns, time-of-day factors, and day-of-week patterns. Uses Random Forest regression trained on historical data.",
                    "operationId": "predictProcessDuration",
                    "parameters": [
                        {
                            "name": "process_def_key",
                            "in": "path",
                            "required": True,
                            "description": "Process definition key",
                            "schema": {
                                "type": "string"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "predicted_duration_hours": {"type": "number"},
                                            "confidence": {"type": "number"},
                                            "confidence_pct": {"type": "number"},
                                            "model_type": {"type": "string"},
                                            "training_instances": {"type": "integer"},
                                            "percentiles": {
                                                "type": "object",
                                                "properties": {
                                                    "p50": {"type": "number"},
                                                    "p75": {"type": "number"},
                                                    "p95": {"type": "number"}
                                                }
                                            },
                                            "message": {"type": "string"},
                                            "timestamp": {"type": "string", "format": "date-time"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/ai/capacity-forecast": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Forecast future capacity needs",
                    "description": "Forecast future capacity requirements based on historical load patterns using time series analysis and trend detection. Identifies peak hours and busiest days to help with capacity planning.",
                    "operationId": "getCapacityForecast",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "forecast": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "day": {"type": "integer"},
                                                        "date": {"type": "string"},
                                                        "predicted_instances": {"type": "integer"},
                                                        "trend": {"type": "string", "enum": ["increasing", "stable", "decreasing"]}
                                                    }
                                                }
                                            },
                                            "growth_rate_per_day": {"type": "number"},
                                            "trend_confidence": {"type": "number"},
                                            "current_avg_daily_load": {"type": "number"},
                                            "patterns": {
                                                "type": "object",
                                                "properties": {
                                                    "peak_hours": {"type": "array"},
                                                    "busiest_days": {"type": "array"}
                                                }
                                            },
                                            "timestamp": {"type": "string", "format": "date-time"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/ai/variable-impact/{process_def_key}": {
                "get": {
                    "tags": ["ai"],
                    "summary": "Analyze process variable impact",
                    "description": "Analyze which process variables correlate with failures or performance issues. Identifies high-impact variables that affect process outcomes, failure rates, and execution times.",
                    "operationId": "getVariableImpact",
                    "parameters": [
                        {
                            "name": "process_def_key",
                            "in": "path",
                            "required": True,
                            "description": "Process definition key",
                            "schema": {
                                "type": "string"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "variable_impacts": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "variable_name": {"type": "string"},
                                                        "variable_type": {"type": "string"},
                                                        "total_instances": {"type": "integer"},
                                                        "failure_rate_pct": {"type": "number"},
                                                        "duration_impact_pct": {"type": "number"},
                                                        "impact_level": {"type": "string", "enum": ["high", "medium", "low"]},
                                                        "recommendation": {"type": "string"},
                                                        "sample_values": {"type": "array", "items": {"type": "string"}}
                                                    }
                                                }
                                            },
                                            "total_analyzed": {"type": "integer"},
                                            "process_def_key": {"type": "string"},
                                            "timestamp": {"type": "string", "format": "date-time"}
                                        }
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
                },
                "AIHealthScore": {
                    "type": "object",
                    "properties": {
                        "overall_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Overall health score (0-100)"
                        },
                        "grade": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D", "F"],
                            "description": "Letter grade representation"
                        },
                        "factors": {
                            "type": "string",
                            "description": "Human-readable factors contributing to score"
                        },
                        "node_scores": {
                            "type": "object",
                            "description": "Individual node health scores"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "AIAnomalies": {
                    "type": "object",
                    "properties": {
                        "anomalies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "process_key": {
                                        "type": "string"
                                    },
                                    "severity": {
                                        "type": "string",
                                        "enum": ["critical", "high", "medium", "low"]
                                    },
                                    "baseline_avg_ms": {
                                        "type": "number"
                                    },
                                    "recent_avg_ms": {
                                        "type": "number"
                                    },
                                    "deviation_pct": {
                                        "type": "number"
                                    },
                                    "z_score": {
                                        "type": "number"
                                    },
                                    "anomaly_types": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            }
                        },
                        "total_analyzed": {
                            "type": "integer"
                        },
                        "detection_window_days": {
                            "type": "integer"
                        },
                        "health_status": {
                            "type": "string"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "AIIncidentPatterns": {
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "incident_type": {
                                        "type": "string"
                                    },
                                    "occurrence_count": {
                                        "type": "integer"
                                    },
                                    "error_message": {
                                        "type": "string"
                                    },
                                    "affected_processes": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    },
                                    "frequency_per_day": {
                                        "type": "number"
                                    }
                                }
                            }
                        },
                        "unique_patterns": {
                            "type": "integer"
                        },
                        "data_source": {
                            "type": "string"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "AIBottlenecks": {
                    "type": "object",
                    "properties": {
                        "bottlenecks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "activity_id": {
                                        "type": "string"
                                    },
                                    "activity_name": {
                                        "type": "string"
                                    },
                                    "process_key": {
                                        "type": "string"
                                    },
                                    "avg_duration_ms": {
                                        "type": "number"
                                    },
                                    "p95_duration_ms": {
                                        "type": "number"
                                    },
                                    "executions": {
                                        "type": "integer"
                                    },
                                    "impact_hours_per_week": {
                                        "type": "number"
                                    }
                                }
                            }
                        },
                        "total_activities_analyzed": {
                            "type": "integer"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "AIJobPredictions": {
                    "type": "object",
                    "properties": {
                        "predictions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "job_type": {
                                        "type": "string"
                                    },
                                    "failure_rate_pct": {
                                        "type": "number"
                                    },
                                    "risk_level": {
                                        "type": "string",
                                        "enum": ["high", "medium", "low"]
                                    },
                                    "failed_count": {
                                        "type": "integer"
                                    },
                                    "total_executions": {
                                        "type": "integer"
                                    },
                                    "recommendation": {
                                        "type": "string"
                                    }
                                }
                            }
                        },
                        "total_jobs_analyzed": {
                            "type": "integer"
                        },
                        "data_source": {
                            "type": "string"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "AINodePerformance": {
                    "type": "object",
                    "properties": {
                        "rankings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_name": {
                                        "type": "string"
                                    },
                                    "rank": {
                                        "type": "integer"
                                    },
                                    "performance_score": {
                                        "type": "number"
                                    },
                                    "recommendation": {
                                        "type": "string"
                                    }
                                }
                            }
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "AIProcessLeaderboard": {
                    "type": "object",
                    "properties": {
                        "leaderboard": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "process_key": {
                                        "type": "string"
                                    },
                                    "grade": {
                                        "type": "string",
                                        "enum": ["A", "B", "C", "D", "F"]
                                    },
                                    "instance_count": {
                                        "type": "integer"
                                    },
                                    "avg_duration_ms": {
                                        "type": "number"
                                    },
                                    "completion_rate_pct": {
                                        "type": "number"
                                    }
                                }
                            }
                        },
                        "total_processes": {
                            "type": "integer"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "AISLAPredictions": {
                    "type": "object",
                    "properties": {
                        "at_risk_tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task_id": {
                                        "type": "string"
                                    },
                                    "task_name": {
                                        "type": "string"
                                    },
                                    "process_instance_id": {
                                        "type": "string"
                                    },
                                    "wait_time_hours": {
                                        "type": "number"
                                    },
                                    "sla_threshold_hours": {
                                        "type": "number"
                                    },
                                    "risk_level": {
                                        "type": "string",
                                        "enum": ["critical", "high", "medium"]
                                    }
                                }
                            }
                        },
                        "total_at_risk": {
                            "type": "integer"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "AIInsights": {
                    "type": "object",
                    "properties": {
                        "health_score": {
                            "$ref": "#/components/schemas/AIHealthScore"
                        },
                        "anomalies": {
                            "$ref": "#/components/schemas/AIAnomalies"
                        },
                        "incidents": {
                            "$ref": "#/components/schemas/AIIncidentPatterns"
                        },
                        "bottlenecks": {
                            "$ref": "#/components/schemas/AIBottlenecks"
                        },
                        "job_failures": {
                            "$ref": "#/components/schemas/AIJobPredictions"
                        },
                        "node_performance": {
                            "$ref": "#/components/schemas/AINodePerformance"
                        },
                        "process_leaderboard": {
                            "$ref": "#/components/schemas/AIProcessLeaderboard"
                        },
                        "sla_predictions": {
                            "$ref": "#/components/schemas/AISLAPredictions"
                        },
                        "recommendations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "priority": {
                                        "type": "string",
                                        "enum": ["critical", "high", "medium", "low"]
                                    },
                                    "message": {
                                        "type": "string"
                                    },
                                    "action": {
                                        "type": "string"
                                    },
                                    "category": {
                                        "type": "string"
                                    }
                                }
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


