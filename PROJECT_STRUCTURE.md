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
│   └── database_service.py # Database metrics
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
└────────────┬────────────────────────┘
             │ HTTP/JSON
             ▼
┌─────────────────────────────────────┐
│      Routes (HTTP Handlers)         │
│  main.py, api.py, metrics.py        │
└────────────┬────────────────────────┘
             │ Function calls
             ▼
┌─────────────────────────────────────┐
│     Services (Business Logic)       │
│  camunda_service, database_service  │
└─────┬──────────────────┬────────────┘
      │                  │
      │ DB queries       │ REST API calls
      ▼                  ▼
┌──────────────┐   ┌──────────────────┐
│  PostgreSQL  │   │  Camunda Nodes   │
│   Database   │   │   (REST API)     │
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

