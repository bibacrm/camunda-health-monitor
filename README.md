# Camunda Health Monitor

A lightweight, real-time monitoring dashboard for Camunda 7 based BPM Platform clusters. Monitor your process engines, track performance metrics, and gain insights into your workflow automation with a modern, responsive interface.

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
[![Docker Hub](https://img.shields.io/docker/v/champabpmn/camunda-health-monitor?label=docker&logo=docker)](https://hub.docker.com/r/champabpmn/camunda-health-monitor) [![Docker Pulls](https://img.shields.io/docker/pulls/champabpmn/camunda-health-monitor)](https://hub.docker.com/r/champabpmn/camunda-health-monitor)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Camunda](https://img.shields.io/badge/camunda-7.x-orange)
![Flask](https://img.shields.io/badge/flask-3.0-green)

## 🎥 Video Demo

Watch the feature demonstration on YouTube(enterprise version, but it is 80% the same):

[![Camunda Health Monitor Demo](https://img.youtube.com/vi/WFQRxpmjRrE/0.jpg)](https://youtu.be/WFQRxpmjRrE)

[▶️ Watch Demo: Champa Camunda 7 Health Monitor](https://youtu.be/WFQRxpmjRrE)

## 📸 Screenshots

### Full Dashboard View
![Dashboard Overview](docs/images/dashboard-full.png)
*Real-time cluster monitoring with comprehensive metrics*

### Light Theme
![Light Theme](docs/images/dashboard-light.png)
*Clean, professional light mode interface*

### Node Status Monitoring
![Node Down Example](docs/images/node-down.png)
*Immediate visual feedback when nodes become unavailable*

---

## Quick Start with Docker

https://hub.docker.com/r/champabpmn/camunda-health-monitor

1. Create a `.env` file with your configuration (example for Camunda 2 nodes cluster):
```bash
# Database Configuration (PostgreSQL)
DB_NAME=PUT_YOUR_CAMUNDA_DB_NAME_HERE
DB_USER=PUT_YOUR_CAMUNDA_DB_USERNAME_HERE
DB_PASSWORD=PUT_YOUR_CAMUNDA_DB_PASS_HERE
DB_HOST=PUT_YOUR_CAMUNDA_DB_HOSTNAME_OR_IP_ADDRESS_HERE
DB_PORT=5432

# Camunda Node 1 (Required)
CAMUNDA_NODE_1_NAME=node1
CAMUNDA_NODE_1_URL=http://PUT_YOUR_CAMUNDA_BPM_1_NODE_HOST_HERE/engine-rest

# Camunda Node 2 (Optional)
CAMUNDA_NODE_2_NAME=node2
CAMUNDA_NODE_2_URL=http://PUT_YOUR_CAMUNDA_BPM_2_NODE_HOST_HERE/engine-rest

# Camunda Node 3 (Optional)
# CAMUNDA_NODE_3_NAME=node3
# CAMUNDA_NODE_3_URL=http://PUT_YOUR_CAMUNDA_BPM_3_NODE_HOST_HERE/engine-rest

# Camunda API Authentication (if enabled)
CAMUNDA_API_USER=
CAMUNDA_API_PASSWORD=

# Optional: JMX/Micrometer Exporter Endpoints
JMX_NODE_1_URL=http://PUT_YOUR_CAMUNDA_BPM_1_NODE_JVM_METRICS_HOST_HERE/metrics
JMX_NODE_2_URL=http://PUT_YOUR_CAMUNDA_BPM_2_NODE_JVM_METRICS_HOST_HERE/metrics

# JVM Metrics Source: 'jmx' or 'micrometer'(from e.g. Quarkus based installation)
JVM_METRICS_SOURCE=jmx

# Stuck Instance Detection (in days)
STUCK_INSTANCE_DAYS=7

# Application Settings
PORT=5000
DEBUG=false
JSON_LOGGING=false
SSL_VERIFY=false
```
2. Run in console:
```bash
docker pull champabpmn/camunda-health-monitor:latest
docker run -d -p 5000:5000 --env-file .env champa_bpmn/camunda-health-monitor:latest
```

3. Access dashboard at http://localhost:5000

---

## 🌟 Features

### Real-Time Cluster Monitoring
- **Multi-node cluster support** - Monitor all your Camunda nodes from a single dashboard
- **Engine health status** - Track node availability and response times
- **Database connectivity** - Monitor PostgreSQL connection health and latency

### Comprehensive Metrics
- **Process Instances** - Active instances, user tasks, and external tasks
- **Job Execution** - Job executor throughput, failed jobs, and execution rates
- **Incidents** - Real-time incident tracking and error monitoring
- **JVM Metrics** - Heap memory, GC statistics, CPU load, and thread counts (requires JMX Exporter)

### Performance Insights
- **Node-level metrics** - Individual node performance and workload distribution
- **Job acquisition rates** - Success rates and rejection tracking
- **Response time tracking** - Node latency monitoring
- **Database analytics** - Storage usage, slow queries, and archivable instances

### Modern UI
- **Dark mode support** - Easy on the eyes for long monitoring sessions
- **Responsive design** - Works on desktop, tablet, and mobile
- **Auto-refresh** - Configurable automatic data refresh
- **Lazy loading** - Fast initial load with on-demand data fetching

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL database (Camunda's backend)
- Camunda 7.x running with REST API enabled
- JMX Exporter or Micrometer based metrics API for advanced JVM metrics

## Installation

### 1. Install

```bash
git clone https://github.com/bibacrm/camunda-health-monitor.git
cd camunda-health-monitor
pip install -r requirements.txt
```

### 2. Configure

Create `.env` file (example for Camunda 2 nodes cluster):

```env
# Database Configuration (PostgreSQL)
DB_NAME=PUT_YOUR_CAMUNDA_DB_NAME_HERE
DB_USER=PUT_YOUR_CAMUNDA_DB_USERNAME_HERE
DB_PASSWORD=PUT_YOUR_CAMUNDA_DB_PASS_HERE
DB_HOST=PUT_YOUR_CAMUNDA_DB_HOSTNAME_OR_IP_ADDRESS_HERE
DB_PORT=5432

# Camunda Node 1 (Required)
CAMUNDA_NODE_1_NAME=node1
CAMUNDA_NODE_1_URL=http://PUT_YOUR_CAMUNDA_BPM_1_NODE_HOST_HERE/engine-rest

# Camunda Node 2 (Optional)
CAMUNDA_NODE_2_NAME=node2
CAMUNDA_NODE_2_URL=http://PUT_YOUR_CAMUNDA_BPM_2_NODE_HOST_HERE/engine-rest

# Camunda Node 3 (Optional)
# CAMUNDA_NODE_3_NAME=node3
# CAMUNDA_NODE_3_URL=http://PUT_YOUR_CAMUNDA_BPM_3_NODE_HOST_HERE/engine-rest

# Camunda API Authentication (if enabled)
CAMUNDA_API_USER=
CAMUNDA_API_PASSWORD=

# Optional: JMX/Micrometer Exporter Endpoints
JMX_NODE_1_URL=http://PUT_YOUR_CAMUNDA_BPM_1_NODE_JVM_METRICS_HOST_HERE/metrics
JMX_NODE_2_URL=http://PUT_YOUR_CAMUNDA_BPM_2_NODE_JVM_METRICS_HOST_HERE/metrics

# JVM Metrics Source: 'jmx' or 'micrometer'(from e.g. Quarkus based installation)
JVM_METRICS_SOURCE=jmx

# Stuck Instance Detection (in days)
STUCK_INSTANCE_DAYS=7

# Application Settings
PORT=5000
DEBUG=false
JSON_LOGGING=false
SSL_VERIFY=false
```

### 3. Run

```bash
# Development
python app.py

# Production
gunicorn -c gunicorn.conf.py app:app
```

Access at `http://localhost:5000`

## Docker

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or build directly
docker build -t camunda-health-monitor .
docker run -p 5000:5000 --env-file .env camunda-health-monitor
```

## Kubernetes

```bash
kubectl apply -f kubernetes-deployment.yaml
```

Edit ConfigMap and Secrets in `kubernetes-deployment.yaml` for your environment.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Dashboard UI |
| `/api/health` | Cluster health JSON |
| `/api/docs` | Swagger documentation |
| `/metrics` | Prometheus metrics |
| `/health/ready` | Readiness probe |
| `/health/live` | Liveness probe |

## Configuration

All configuration via environment variables. See `.env.example` for full reference.

**Key Variables:**
- `DB_*` - PostgreSQL connection
- `CAMUNDA_NODE_*` - Camunda REST API endpoints (supports multiple nodes)
- `JMX_NODE_*` - JMX/Micrometer endpoints (optional)
- `CAMUNDA_API_USER/PASSWORD` - Basic auth (optional)
- `JVM_METRICS_SOURCE` - `jmx` or `micrometer`
- `SSL_VERIFY` - `true`/`false` for HTTPS verification

## Architecture

```
├── app.py              # Flask application factory
├── config.py           # Configuration management
├── wsgi.py             # Production WSGI entry point
├── routes/             # HTTP endpoints (blueprints)
│   ├── main.py         # Dashboard
│   ├── api.py          # API endpoints
│   └── metrics.py      # Prometheus & health
├── services/           # Business logic
│   ├── camunda_service.py
│   └── database_service.py
└── helpers/            # Utilities
    ├── db_helper.py
    ├── health_checks.py
    └── error_handler.py
```

## Production Deployment

**Gunicorn** (recommended):
```bash
gunicorn -c gunicorn.conf.py app:app
```

**Configuration** in `gunicorn.conf.py`:
- Workers: CPU cores × 2 + 1
- Timeout: 120s
- Graceful shutdown: 30s

**Database Pool**:
- Min connections: 1
- Max connections: 20
- Timeout: 5s

## Monitoring

**Prometheus**:
```yaml
scrape_configs:
  - job_name: 'camunda-monitor'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
```

**Kubernetes Probes**:
- Startup: `/health/startup` (max 150s)
- Liveness: `/health/live` (every 10s)
- Readiness: `/health/ready` (every 5s)

## 📈 Metrics Overview

### Cluster Status
- Total nodes and running nodes
- Engine version information
- System health alerts

### Database Health
- Connection latency
- Active connections
- Connection pool utilization
- Storage usage by table
- Slow query analysis

### Process Metrics
- Active process instances
- User tasks waiting
- External tasks in queue
- Open incidents
- Stuck instances (configurable threshold)
- Pending messages and signals

### Job Executor
- Total jobs in queue
- Job execution throughput (jobs/min)
- Failed jobs (no retries left)
- Executable jobs ready to run

### Node-Specific Metrics
- Response times
- Job acquisition rates
- Execution success rates
- Workload distribution
- JVM health (heap, GC, threads, CPU)

## 🎨 UI Features

### Dark Mode
Toggle between light and dark themes for comfortable monitoring in any environment.

### Auto-Refresh
Enable automatic refresh to keep metrics up-to-date (default: 30 seconds).

### Mobile-Friendly
Responsive design ensures full functionality on desktop, tablet, and mobile devices.

### Lazy Loading
Individual metric cards load on-demand for faster initial page load.

## 🔒 Security Considerations

This is a monitoring tool that requires read access to:
- Camunda REST API endpoints
- PostgreSQL database

**Important Security Practices:**

- Use read-only database credentials when possible
- Enable authentication on Camunda REST API in production
- Run behind a reverse proxy with HTTPS in production
- Consider network segmentation for database access
- Use environment variables for all sensitive configuration
- Never commit `.env` files to version control

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE.md](LICENSE.md) for details.

**Voluntary Support**: While this software is free and open source, voluntary contributions help maintain and improve this project. See [LICENSING_FAQ.md](LICENSING_FAQ.md) and [LICENSE.md](LICENSE.md) for more information about supporting the project.

## 🙏 Acknowledgments

- Built for the [Camunda 7 BPM Platform](https://camunda.com/platform/camunda-bpm/)
- UI powered by [Alpine.js](https://alpinejs.dev/) and [Tailwind CSS](https://tailwindcss.com/)
- Icons by [Lucide](https://lucide.dev/)

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/bibacrm/camunda-health-monitor/issues)
- **Enterprise Solutions**: [Champa Intelligence](https://champa-bpmn.com)
- **Email**: info@champa-bpmn.com

---

**Developed by [Champa Intelligence](https://champa-bpmn.com)** - Enterprise BPM Solutions

For advanced features, professional support, and enterprise deployments, visit [champa-bpmn.com](https://champa-bpmn.com).
