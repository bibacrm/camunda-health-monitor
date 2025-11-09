# Project Structure & Architecture

## 📁 File Organization

```
camunda-health-monitor/
│
├── 📄 app.py                     # Main Flask application (500 lines)
├── 📁 templates/
│   └── 📄 index.html            # Dashboard UI with embedded JS/CSS
│
├── 📄 requirements.txt           # Python dependencies (4 packages)
├── 📄 .env.example              # Configuration template
│
├── 🐳 Dockerfile                # Container build configuration
├── 🐳 docker-compose.yml        # Docker orchestration
│
├── 📄 README.md                 # Project documentation
├── 📄 CONTRIBUTING.md           # Contribution guidelines
├── 📄 LICENSE                   # MIT License
└── 📄 PROJECT_STRUCTURE.md      # This file
```

## 🏗️ Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  HTML/CSS   │  │  Alpine.js   │  │  Tailwind    │  │
│  │  Dashboard  │  │  Reactivity  │  │  Styling     │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP/JSON
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Flask Application (app.py)                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Routes:                                        │   │
│  │  • GET  /              → Dashboard HTML        │   │
│  │  • GET  /api/health    → Full health JSON      │   │
│  │  • GET  /api/metrics/* → Individual metrics    │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Core Functions:                                │   │
│  │  • collect_engine_health()                      │   │
│  │  • fetch_node_data()                            │   │
│  │  • collect_jmx_metrics()                        │   │
│  │  • execute_query()                              │   │
│  └─────────────────────────────────────────────────┘   │
└───────────┬─────────────────────┬───────────────────────┘
            │                     │
            ▼                     ▼
   ┌────────────────┐    ┌────────────────────┐
   │   PostgreSQL   │    │  Camunda REST API  │
   │    Database    │    │   (Multiple Nodes) │
   └────────────────┘    └────────────────────┘
```

### Data Flow

```
1. Browser Request
   └─→ Flask Route Handler
       ├─→ Database Queries (parallel)
       │   └─→ PostgreSQL
       │       └─→ Camunda tables (act_*)
       │
       ├─→ REST API Calls (parallel)
       │   └─→ Each Camunda Node
       │       ├─→ /engine
       │       ├─→ /metrics
       │       └─→ /process-instance/count
       │
       ├─→ JMX Metrics (parallel)
       │   └─→ JMX Exporter/Micrometer API endpoints
       │
       └─→ Aggregate & Format
           └─→ Return JSON
               └─→ Browser renders with Alpine.js
```

## 🧩 Component Details

### Backend Components (app.py)

#### 1. Configuration Management
```python
# Loads from environment variables
- Database connection parameters
- Camunda node URLs (dynamic multi-node support)
- JMX endpoint URLs
- Application settings
```

#### 2. Database Layer
```python
def get_db_connection():
    """Create PostgreSQL connection"""

def execute_query(query, params=None):
    """Execute SQL and return results as dict"""
```

#### 3. JMX Metrics Collection
```python
def parse_prometheus_metrics(text):
    """Parse Prometheus format metrics"""

def collect_jmx_metrics():
    """Parallel fetch from all JMX endpoints"""

def extract_jvm_health_metrics(data):
    """Extract key JVM indicators"""
```

#### 4. Node Health Collection
```python
def fetch_node_data(node_name, node_url, jmx_data):
    """
    For each Camunda node:
    - Check engine status
    - Get response time
    - Fetch metrics from REST API
    - Combine with JVM data
    """
```

#### 5. Main Collector
```python
def collect_engine_health():
    """
    Orchestrates all collection:
    1. Parallel JMX collection
    2. Parallel node data collection
    3. Database health check
    4. Aggregate cluster totals
    5. Return complete health picture
    """
```

### Frontend Components (index.html)

#### 1. Alpine.js Application State
```javascript
function healthMonitor() {
    return {
        data: {...},              // Health metrics
        darkMode: false,          // Theme preference
        autoRefresh: false,       // Auto-refresh toggle
        isLoading: false,         // Loading state
        
        // Methods
        refreshData(),            // Fetch latest data
        toggleDarkMode(),         // Theme switcher
        formatNumber(),           // Number formatting
        formatDuration()          // Duration formatting
    }
}
```

#### 2. UI Sections

**Header**
- Application title
- Dark mode toggle
- Auto-refresh toggle
- Manual refresh button

**System Status Cards**
- Engine status (nodes running)
- Database health (latency, connections)
- Active instances (processes, tasks)
- Incidents (errors, failed jobs)

**Cluster Nodes Grid**
- Per-node health cards
- JVM metrics (heap, CPU, threads)
- Response times
- Job execution stats

**Issues Alert**
- Dynamic alert when problems detected
- Lists all cluster issues

#### 3. Styling
- Tailwind CSS utility classes
- Dark mode support (via `dark:` prefix)
- Responsive design (sm, md, lg breakpoints)
- Custom scrollbar styling

## 🔄 Request/Response Cycle

### Dashboard Load (`GET /`)

```
1. Browser → Flask
2. Flask calls collect_engine_health()
3. Parallel execution:
   ├─ JMX metrics from all nodes (ThreadPoolExecutor)
   ├─ Health check each node (ThreadPoolExecutor)
   ├─ Database connectivity test
   └─ Shared state from first available node
4. Aggregate results
5. Render template with data
6. Return HTML
7. Browser loads → Alpine.js initializes
8. Lucide renders icons
```

### API Refresh (`GET /api/health`)

```
1. Browser (Alpine.js) → Flask /api/health
2. Flask calls collect_engine_health()
3. [Same parallel collection as above]
4. Return JSON response
5. Alpine.js updates reactive data
6. UI re-renders automatically
7. Lucide re-renders icons
```

### Individual Metric (`GET /api/metrics/stuck-instances`)

```
1. Browser → Flask /api/metrics/stuck-instances
2. Flask executes specific database query
3. Return JSON with single metric
4. Browser updates that specific UI element
```

## 🚀 Deployment Modes

### 1. Direct Python Execution
```bash
python app.py
# Runs on http://localhost:5000
```

### 2. Docker Container
```bash
docker build -t camunda-health-monitor .
docker run -p 5000:5000 --env-file .env camunda-health-monitor
```

### 3. Docker Compose
```bash
docker-compose up
# Manages configuration and restart policy
```

### 4. Production (Behind Reverse Proxy)
```
Internet → Nginx/Apache/Traefik → Flask Application
         (HTTPS, rate limiting)
```

## 🔐 Security Considerations

### Current Implementation

✅ **What's Included:**
- Environment-based configuration (no hardcoded credentials)
- Optional Camunda API authentication support
- Read-only database operations
- CORS not enabled (same-origin only)
- No user authentication (monitoring tool only)

⚠️ **Production Recommendations:**
- Run behind reverse proxy with HTTPS
- Use read-only database user
- Enable Camunda API authentication
- Restrict network access to monitoring team
- Consider adding basic auth at reverse proxy level
- Use Docker secrets for sensitive values

## 📊 Metrics Collected

### From Database (PostgreSQL)
- Active process instances
- User tasks count
- External tasks count
- Open incidents
- Total jobs
- Failed jobs count
- Stuck instance detection

### From Camunda REST API
- Engine version
- Node availability
- Response times
- Process definitions
- DMN definitions
- Deployment count

### From JMX Exporter
- Heap memory usage & limits
- GC statistics
- Thread count
- CPU load
- File descriptor usage

## 🎨 UI/UX Features

### Responsive Design
- Mobile-first approach
- Breakpoints: sm (640px), md (768px), lg (1024px)
- Collapsible sections on small screens
- Touch-friendly controls

### Dark Mode
- Persistent preference (localStorage)
- System-wide color scheme
- Smooth transitions
- Proper contrast ratios

### Performance
- Lazy loading not implemented (single page)
- Efficient re-rendering with Alpine.js
- Minimal external dependencies
- CDN-served libraries for fast loading

### Accessibility
- Semantic HTML elements
- Icon labels for screen readers
- Keyboard navigation support
- Color contrast compliance

## 🔧 Configuration Options

### Required Settings
```env
DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
CAMUNDA_NODE_1_NAME, CAMUNDA_NODE_1_URL
```

### Optional Settings
```env
CAMUNDA_API_USER, CAMUNDA_API_PASSWORD    # If auth enabled
JMX_NODE_1_URL, JMX_NODE_2_URL...        # For JVM metrics
JVM_METRICS_SOURCE                        # jmx or micrometer
STUCK_INSTANCE_DAYS                       # Detection threshold
PORT                                      # Application port
DEBUG                                     # Debug mode flag
```

### Multi-Node Configuration
```env
# Add as many nodes as needed
CAMUNDA_NODE_1_NAME=prod-1
CAMUNDA_NODE_1_URL=http://prod1:8080/engine-rest

CAMUNDA_NODE_2_NAME=prod-2
CAMUNDA_NODE_2_URL=http://prod2:8080/engine-rest

CAMUNDA_NODE_3_NAME=prod-3
CAMUNDA_NODE_3_URL=http://prod3:8080/engine-rest
```
