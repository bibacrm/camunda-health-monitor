# Camunda Health Monitor

A lightweight, real-time monitoring dashboard for Camunda 7 based BPM Platform clusters. Monitor your process engines, track performance metrics, and gain insights into your workflow automation with a modern, responsive interface.

![License](https://img.shields.io/badge/license-Custom-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Camunda](https://img.shields.io/badge/camunda-7.x-orange)
![Flask](https://img.shields.io/badge/flask-3.0-green)

## 🎥 Video Demo

Watch the feature demonstration on YouTube(enterprise version, but it is 70% the same):

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

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/bibacrm/camunda-health-monitor.git
cd camunda-health-monitor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your Environment

Create a `.env` file in the project root:

```env
# Database Configuration
DB_NAME=camunda
DB_USER=camunda
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# Camunda Nodes
CAMUNDA_NODE_1_NAME=node1
CAMUNDA_NODE_1_URL=http://localhost:8080/engine-rest

CAMUNDA_NODE_2_NAME=node2
CAMUNDA_NODE_2_URL=http://localhost:8081/engine-rest

# Optional: Camunda API Authentication
CAMUNDA_API_USER=
CAMUNDA_API_PASSWORD=

# Optional: JMX Exporter Endpoints
JMX_NODE_1_URL=http://localhost:8080/metrics
JMX_NODE_2_URL=http://localhost:8081/metrics

# Optional: JVM Metrics Source (jmx or micrometer)
JVM_METRICS_SOURCE=jmx

# Configuration
STUCK_INSTANCE_DAYS=7
PORT=5000
DEBUG=false
```

### 4. Run the Application

**Option 1: Development Mode (Flask)**

```bash
python app.py
```

**Option 2: Production Mode (Gunicorn)**

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

The dashboard will be available at `http://localhost:5000`

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env

# Start the application
docker-compose up -d
```

### Using Docker Directly

```bash
# Build the image
docker build -t camunda-health-monitor .

# Run the container
docker run -d \
  --name camunda-health-monitor \
  -p 5000:5000 \
  --env-file .env \
  camunda-health-monitor
```

### Health Check

```bash
curl http://localhost:5000/api/health
```

## 📊 API Endpoints

### REST API

- `GET /` - Main dashboard UI
- `GET /api/health` - Full health metrics (JSON)
- `GET /api/metrics/stuck-instances` - Stuck instances count
- `GET /api/metrics/pending-messages` - Pending message subscriptions
- `GET /api/metrics/pending-signals` - Pending signal subscriptions
- `GET /api/metrics/job-throughput` - Job execution rate
- `GET /api/metrics/database` - Database storage and performance metrics

### Prometheus Metrics

Export metrics for Prometheus/Grafana:

```bash
curl http://localhost:5000/metrics
```

Metrics include:
- `camunda_active_instances` - Active process instances
- `camunda_incidents` - Open incidents
- `camunda_node_status` - Node availability (per node)
- `camunda_jvm_heap_utilization_percent` - JVM heap usage (per node)
- `camunda_db_latency_ms` - Database query latency
- And many more...

## 🔧 Configuration

### Multiple Nodes

To monitor multiple Camunda nodes, add additional node configurations:

```env
CAMUNDA_NODE_1_NAME=production-1
CAMUNDA_NODE_1_URL=http://prod1.example.com:8080/engine-rest

CAMUNDA_NODE_2_NAME=production-2
CAMUNDA_NODE_2_URL=http://prod2.example.com:8080/engine-rest

CAMUNDA_NODE_3_NAME=production-3
CAMUNDA_NODE_3_URL=http://prod3.example.com:8080/engine-rest
```

### JMX Exporter Setup

For detailed JVM metrics, configure JMX Exporter on your Camunda nodes:

1. Download [Prometheus JMX Exporter](https://github.com/prometheus/jmx_exporter)
2. Add to your Camunda JVM startup:
   ```bash
   -javaagent:/path/to/jmx_prometheus_javaagent.jar=8080:/path/to/config.yaml
   ```
3. Add endpoints to your `.env`:
   ```env
   JMX_NODE_1_URL=http://localhost:8080/metrics
   ```

### Gunicorn Production Configuration

Create `gunicorn.conf.py`:

```python
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
errorlog = "logs/gunicorn-error.log"
accesslog = "logs/gunicorn-access.log"
loglevel = "info"
```

Run with:
```bash
gunicorn -c gunicorn.conf.py app:app
```

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

### Example Nginx Reverse Proxy

```nginx
server {
    listen 443 ssl;
    server_name monitor.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under a custom non-commercial license. See [LICENSE.md](LICENSE.md) for details.

**Summary:**
- ✅ Free for educational, non-profit, and personal use
- ❌ Commercial use requires a separate license
- 📧 Contact: info@champa-bpmn.com

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