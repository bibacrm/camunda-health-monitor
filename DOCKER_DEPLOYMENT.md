# Docker Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)

```bash
# 1. Clone or navigate to the project directory
cd camunda-health-monitor

# 2. Create/edit .env file with your configuration
cp .env.example .env
nano .env

# 3. Build and run
docker-compose up -d

# 4. Check logs
docker-compose logs -f

# 5. Access the dashboard
# http://localhost:5000
```

### Using Docker (Manual)

```bash
# Build the image
docker build -t camunda-health-monitor:latest .

# Run the container
docker run -d \
  --name camunda-health-monitor \
  -p 5000:5000 \
  --env-file .env \
  camunda-health-monitor:latest
```

---

## Configuration

### Environment Variables

All configuration is done via environment variables. See `.env.example` for the complete list.

**Required**:
- `DB_NAME` - Database name
- `DB_USER` - Database user
- `DB_PASSWORD` - Database password
- `DB_HOST` - Database host
- `DB_PORT` - Database port
- `CAMUNDA_NODE_1_NAME` - First Camunda node name
- `CAMUNDA_NODE_1_URL` - First Camunda REST API URL

**Optional**:
- `SSL_VERIFY` - SSL certificate verification (default: `false`)
- `JSON_LOGGING` - Enable JSON logging (default: `false`)
- `JVM_METRICS_SOURCE` - JVM metrics source: `jmx` or `micrometer` (default: `jmx`)
- `STUCK_INSTANCE_DAYS` - Days to consider instance stuck (default: `7`)
- `DEBUG` - Debug mode (default: `false`)

---

## Docker Compose Configuration

### Basic Setup

```yaml
version: '3.8'

services:
  health-monitor:
    image: camunda-health-monitor:latest
    container_name: camunda-health-monitor
    ports:
      - "5000:5000"
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

### With Resource Limits

```yaml
services:
  health-monitor:
    # ...existing config...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 256M
```

### With Custom Network

```yaml
services:
  health-monitor:
    # ...existing config...
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge
```

---

## Building the Image

### Standard Build

```bash
docker build -t camunda-health-monitor:latest .
```

### Build with Custom Tag

```bash
docker build -t camunda-health-monitor:v1.0.0 .
```

### Multi-platform Build

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t camunda-health-monitor:latest \
  --push .
```

---

## Running the Container

### Basic Run

```bash
docker run -d \
  --name camunda-health-monitor \
  -p 5000:5000 \
  --env-file .env \
  camunda-health-monitor:latest
```

### With Volume Mounts

```bash
docker run -d \
  --name camunda-health-monitor \
  -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  camunda-health-monitor:latest
```

### With Host Network (for localhost Camunda)

```bash
docker run -d \
  --name camunda-health-monitor \
  --network host \
  --env-file .env \
  camunda-health-monitor:latest
```

---

## Health Checks

### Docker Health Check

The image includes a built-in health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1
```

### Check Container Health

```bash
# View health status
docker inspect --format='{{.State.Health.Status}}' camunda-health-monitor

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' camunda-health-monitor
```

---

## Management Commands

### Start/Stop

```bash
# Start
docker-compose start

# Stop
docker-compose stop

# Restart
docker-compose restart
```

### View Logs

```bash
# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f health-monitor
```

### Execute Commands

```bash
# Open shell
docker-compose exec health-monitor /bin/bash

# Run Python command
docker-compose exec health-monitor python -c "print('Hello')"
```

### Update and Restart

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose up -d --build
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs health-monitor

# Check if port is in use
netstat -tuln | grep 5000

# Check configuration
docker-compose config
```

### Database Connection Issues

```bash
# Test database connectivity from container
docker-compose exec health-monitor \
  psql -h ${DB_HOST} -U ${DB_USER} -d ${DB_NAME}
```

### SSL Certificate Issues

If you're using self-signed certificates:

```bash
# Set SSL_VERIFY=false in .env
echo "SSL_VERIFY=false" >> .env

# Restart
docker-compose restart
```

### High Memory Usage

```bash
# Check resource usage
docker stats camunda-health-monitor

# Reduce Gunicorn workers
docker-compose exec health-monitor \
  printenv | grep GUNICORN_WORKERS
```

---

## Production Deployment

### Recommended docker-compose.yml

```yaml
version: '3.8'

services:
  health-monitor:
    image: camunda-health-monitor:latest
    container_name: camunda-health-monitor
    
    ports:
      - "5000:5000"
    
    env_file:
      - .env
    
    volumes:
      - ./logs:/app/logs
    
    restart: always
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 256M
    
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Security Best Practices

1. **Use secrets for sensitive data**:
   ```yaml
   secrets:
     - db_password
   
   services:
     health-monitor:
       secrets:
         - db_password
       environment:
         - DB_PASSWORD_FILE=/run/secrets/db_password
   ```

2. **Run as non-root** (already configured in Dockerfile)

3. **Use read-only filesystem** (if possible):
   ```yaml
   services:
     health-monitor:
       read_only: true
       tmpfs:
         - /tmp
         - /app/logs
   ```

4. **Enable JSON logging** for better log aggregation:
   ```bash
   JSON_LOGGING=true
   ```

---

## Kubernetes Deployment

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: camunda-health-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: camunda-health-monitor
  template:
    metadata:
      labels:
        app: camunda-health-monitor
    spec:
      containers:
      - name: health-monitor
        image: camunda-health-monitor:latest
        ports:
        - containerPort: 5000
        envFrom:
        - configMapRef:
            name: health-monitor-config
        - secretRef:
            name: health-monitor-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          limits:
            cpu: "2"
            memory: "1Gi"
          requests:
            cpu: "500m"
            memory: "256Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: camunda-health-monitor
spec:
  selector:
    app: camunda-health-monitor
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

---

## Monitoring

### Prometheus Integration

The application exposes metrics at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'camunda-health-monitor'
    static_configs:
      - targets: ['health-monitor:5000']
    metrics_path: '/metrics'
```

### Log Aggregation

With JSON logging enabled:

```yaml
# filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
    - add_docker_metadata: ~
    - decode_json_fields:
        fields: ["message"]
        target: ""
        overwrite_keys: true
```

---

## Backup and Recovery

### Backup Logs

```bash
# Create backup
docker cp camunda-health-monitor:/app/logs ./logs-backup-$(date +%Y%m%d)

# Or with volume
tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/
```

### Export Configuration

```bash
# Export environment variables
docker-compose exec health-monitor printenv > config-backup.txt
```

---

## Performance Tuning

### Gunicorn Workers

Adjust based on CPU cores:

```bash
# Set in .env or docker-compose.yml
GUNICORN_WORKERS=5
```

### Database Connection Pool

The application uses connection pooling (1-20 connections). Monitor usage:

```bash
# Check pool size
curl http://localhost:5000/health | jq '.checks.database.pool_size'
```

### Memory Optimization

```yaml
deploy:
  resources:
    limits:
      memory: 512M  # Adjust based on monitoring
```

---

## Upgrading

### Zero-Downtime Upgrade

```bash
# Pull latest code
git pull

# Build new image
docker build -t camunda-health-monitor:new .

# Test new image
docker run -d --name test-monitor \
  --env-file .env \
  -p 5001:5000 \
  camunda-health-monitor:new

# If tests pass, update production
docker-compose down
docker-compose up -d --build
```

---

*Last updated: 2025-11-10*

