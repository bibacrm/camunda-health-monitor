# Project Structure & Architecture

## Directory Structure

```
camunda-health-monitor/
├── app.py                    # Application factory & initialization
├── config.py                 # Configuration management
├── wsgi.py                   # Production WSGI entry point
├── gunicorn.conf.py         # Gunicorn configuration
├── requirements.txt         # Python dependencies
├── .env.example             # Configuration template
│
├── routes/                  # HTTP endpoints (Flask Blueprints)
│   ├── __init__.py         # Blueprint registration
│   ├── main.py             # Dashboard route
│   ├── api.py              # API endpoints
│   └── metrics.py          # Prometheus & health checks
│
├── services/               # Business logic layer
│   ├── __init__.py
│   ├── camunda_service.py  # Camunda health collection
│   ├── database_service.py # Database metrics
│   └── ai_service.py       # AI/ML analytics
│
├── helpers/                # Utility modules
│   ├── __init__.py
│   ├── db_helper.py        # Database connection pooling
│   ├── error_handler.py    # Error handling & circuit breakers
│   ├── health_checks.py    # Kubernetes health probes
│   ├── swagger_config.py   # API documentation
│   └── shutdown_handler.py # Graceful shutdown
│
├── templates/              # HTML templates
│   └── index.html         # Dashboard UI (Alpine.js + Tailwind)
│
├── Dockerfile             # Container build
├── docker-compose.yml     # Docker orchestration
└── kubernetes-deployment.yaml  # Kubernetes manifests
```

## Architecture

### Application Layers

```
┌─────────────────────────────────────┐
│         Browser (UI)                │
│  Alpine.js + Tailwind CSS           │
│  - Main Dashboard                   │
│  - AI Intelligence Layer            │
└────────────┬────────────────────────┘
             │ HTTP/JSON
             ▼
┌─────────────────────────────────────┐
│      Routes (HTTP Handlers)         │
│  main.py, api.py, metrics.py        │
│  - /api/ai/* endpoints              │
│  - Prometheus AI metrics            │
└────────────┬────────────────────────┘
             │ Function calls
             ▼
┌──────────────────────────────────────┐
│     Services (Business Logic)        │
│  - camunda_service                   │
│  - database_service                  │
│  - ai_service                        │
│    • Anomaly Detection               │
│    • Pattern Recognition             │
│    • Predictive Analytics            │
│    • Performance scoring and rankings│
│    • Bottleneck identification       │
│    • Health score calculations       │
└─────┬──────────────────┬─────────────┘
      │                  │
      │ DB queries       │ REST API calls
      ▼                  ▼
┌──────────────┐   ┌──────────────────┐
│  PostgreSQL  │   │  Camunda Nodes   │
│   Database   │   │   (REST API)     │
│  ACT_HI_*    │   │   + JMX/Metrics  │
│  ACT_RU_*    │   │                  │
└──────────────┘   └──────────────────┘
```

### Key Components

**Application Factory** (`app.py`)
- Creates Flask application
- Loads configuration
- Initializes database pool
- Registers blueprints
- Sets up health probes

**Configuration** (`config.py`)
- Loads environment variables
- Validates required settings
- Provides config objects (Dev/Prod/Test)

**Routes** (`routes/`)
- Thin HTTP handlers
- Delegate to services
- Return JSON or render templates

**Services** (`services/`)
- Business logic
- Parallel data collection
- Metrics aggregation
- Error handling with circuit breakers
- **AI/ML Analytics**:
  - Anomaly detection using statistical analysis
  - Incident pattern recognition with clustering
  - Predictive analytics for job failures and SLA breaches
  - Performance scoring and rankings
  - Bottleneck identification
  - Health score calculations

**Helpers** (`helpers/`)
- Database connection pooling
- Circuit breakers
- Health check registry
- Graceful shutdown
- API documentation

## Data Flow

### Dashboard Load

```
Browser → main.py
  → camunda_service.collect_engine_health()
    ├─→ Parallel: Fetch from all Camunda nodes (REST API)
    ├─→ Parallel: Fetch JMX metrics (if configured)
    ├─→ Parallel: Query PostgreSQL
    └─→ Aggregate results
  ← Return JSON
← Render template
```

### API Call (`/api/health`)

```
Browser → api.py
  → camunda_service.collect_engine_health()
  ← Return JSON directly
```

### Health Probe (`/health/ready`)

```
Kubernetes → metrics.py
  → health_checks.check_readiness()
    ├─→ Test database connection
    ├─→ Test Camunda node reachability
    └─→ Check circuit breaker states
  ← Return 200 (ready) or 503 (not ready)
```

### AI Insights (`/api/ai/insights`)

```
Browser → api.py → ai_service.get_ai_insights()
  → Parallel execution of all AI features:
    ├─→ get_cluster_health_score(cluster_data, db_metrics)
    │   └─→ Calculate composite score (no DB queries)
    ├─→ detect_process_anomalies(lookback_days=7)
    │   └─→ Query ACT_HI_PROCINST (50K rows max)
    │   └─→ Statistical Z-score analysis
    ├─→ analyze_incident_patterns(lookback_days=30)
    │   └─→ Query ACT_HI_INCIDENT or ACT_RU_JOB
    │   └─→ Pattern clustering and grouping
    ├─→ identify_bottlenecks(lookback_days=7)
    │   └─→ Query ACT_HI_ACTINST (50K rows max)
    │   └─→ Percentile calculations (P95, P99)
    ├─→ predict_job_failures(lookback_days=7)
    │   └─→ Query ACT_RU_JOB with exceptions
    │   └─→ Failure rate analysis
    ├─→ analyze_node_performance(cluster_nodes)
    │   └─→ JVM metrics analysis (no DB queries)
    ├─→ get_process_leaderboard(lookback_days=30)
    │   └─→ Query ACT_HI_PROCINST with aggregations
    └─→ predict_sla_breaches(threshold_hours=24)
        └─→ Query ACT_RU_TASK
        └─→ Calculate wait times
  ← Aggregate all results
  ← Generate recommendations
← Return comprehensive JSON
```

### Prometheus AI Metrics (`/metrics`)

```
Prometheus → metrics.py → _collect_ai_metrics_for_prometheus()
  → Fast aggregation queries (NO row fetching):
    ├─→ Health Score (uses existing cluster_data, 0 queries)
    ├─→ Anomaly Count (COUNT query with avg > 1 hour)
    ├─→ Incident Patterns (COUNT DISTINCT incident types)
    ├─→ Bottleneck Count (COUNT with avg > 10 seconds)
    └─→ SLA At-Risk (COUNT tasks created > 80% threshold)
  ← Return metrics in Prometheus format
← Scrape completed
```

## Key Features

### Parallel Processing
- All Camunda nodes queried simultaneously
- JMX metrics fetched in parallel
- Database queries run concurrently
- Total collection time ~2-3 seconds for 4 nodes

### Circuit Breakers
- Protect against cascading failures
- Automatic retry with backoff
- Fail-fast when service is down

### Database Connection Pooling
- 1-20 connection pool
- Automatic connection management
- Thread-safe operations

### Health Probes
- `/health/live` - Container is alive
- `/health/ready` - Ready for traffic
- `/health/startup` - Initialization complete

## Configuration

All configuration via environment variables:

```env
# Database (required)
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# Camunda nodes (at least one required)
CAMUNDA_NODE_1_NAME, CAMUNDA_NODE_1_URL
CAMUNDA_NODE_2_NAME, CAMUNDA_NODE_2_URL...

# Optional: JMX metrics
JMX_NODE_1_URL, JMX_NODE_2_URL...
JVM_METRICS_SOURCE=jmx  # or 'micrometer'

# Optional: Authentication
CAMUNDA_API_USER, CAMUNDA_API_PASSWORD

# Optional: Settings
PORT=5000
DEBUG=false
JSON_LOGGING=false
SSL_VERIFY=false
STUCK_INSTANCE_DAYS=7

# AI/ML Main Configuration
AI_LOOKBACK_DAYS=30                     # Historical analysis window for all AI features
AI_MAX_INSTANCES=50000                  # Max instances to query for AI analysis
AI_MAX_INCIDENTS=1000                   # Max incidents for pattern recognition
AI_MIN_DATA=10                          # Minimum data points for analysis
AI_UI_RESULTS_LIMIT=20                  # Max results to display in UI/API
SLA_THRESHOLD_HOURS=24                  # SLA breach threshold
UI_AUTO_REFRESH_INTERVAL_MS=30000       # Auto-refresh interval
DB_ARCHIVE_THRESHOLD_DAYS=90            # Archive display threshold
```

## Deployment Modes

**Development**
```bash
python app.py
```

**Production (Gunicorn)**
```bash
gunicorn -c gunicorn.conf.py app:app
```

**Docker**
```bash
docker-compose up -d
```

**Kubernetes**
```bash
kubectl apply -f kubernetes-deployment.yaml
```

---

For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md)

