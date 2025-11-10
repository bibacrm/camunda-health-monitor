# Docker Quick Reference

## ⚡ Quick Start

```bash
# 1. Setup
cp .env.example .env
nano .env  # Edit your configuration

# 2. Run
docker-compose up -d

# 3. Check
docker-compose logs -f
```

**Dashboard**: http://localhost:5000

---

## 🚀 Common Commands

### Start/Stop
```bash
docker-compose start      # Start
docker-compose stop       # Stop
docker-compose restart    # Restart
docker-compose down       # Stop and remove
```

### Logs
```bash
docker-compose logs -f              # Follow logs
docker-compose logs --tail=100      # Last 100 lines
docker-compose logs -f health-monitor  # Specific service
```

### Update
```bash
git pull
docker-compose up -d --build
```

### Shell Access
```bash
docker-compose exec health-monitor /bin/bash
```

---

## 🔧 Configuration

### Required Environment Variables

```env
# Database
DB_NAME=camunda
DB_USER=camunda
DB_PASSWORD=your-password
DB_HOST=localhost
DB_PORT=5432

# Camunda Node
CAMUNDA_NODE_1_NAME=node1
CAMUNDA_NODE_1_URL=http://localhost:8080/engine-rest
```

### New Configuration Options

```env
# SSL Certificate Verification
SSL_VERIFY=false  # Set to true for production

# JSON Logging (for ELK/Splunk)
JSON_LOGGING=false

# JVM Metrics
JVM_METRICS_SOURCE=micrometer  # or jmx
```

---

## 🏥 Health Checks

```bash
# Container health
docker inspect --format='{{.State.Health.Status}}' camunda-health-monitor

# Application health
curl http://localhost:5000/health
```

---

## 🐛 Troubleshooting

### Container won't start
```bash
docker-compose logs health-monitor
docker-compose config  # Validate config
```

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "5001:5000"  # Use 5001 instead
```

### Database connection failed
```bash
# Test from container
docker-compose exec health-monitor \
  psql -h $DB_HOST -U $DB_USER -d $DB_NAME
```

### High memory usage
```bash
docker stats camunda-health-monitor

# Reduce workers in .env
GUNICORN_WORKERS=3
```

---

## 📊 Resource Limits

### Current Limits
- CPU: 2 cores max, 0.5 cores reserved
- Memory: 1GB max, 256MB reserved

### Adjust in docker-compose.yml
```yaml
deploy:
  resources:
    limits:
      cpus: '1'      # Reduce for smaller servers
      memory: 512M
```

---

## 🔐 Security

### Run on different port
```yaml
ports:
  - "8080:5000"  # Access on port 8080
```

### Add SSL/TLS (with reverse proxy)
Use nginx or similar in front

### Restrict network access
```yaml
networks:
  - internal-only

networks:
  internal-only:
    internal: true
```

---

## 📦 Build

```bash
# Standard build
docker build -t camunda-health-monitor:latest .

# With custom tag
docker build -t camunda-health-monitor:v1.0.0 .

# Multi-platform
docker buildx build --platform linux/amd64,linux/arm64 -t myrepo/health-monitor:latest .
```

---

## 🎯 Production Checklist

- [ ] SSL_VERIFY=true (if using valid certificates)
- [ ] JSON_LOGGING=true (for log aggregation)
- [ ] Set proper resource limits
- [ ] Configure log rotation
- [ ] Set up monitoring (Prometheus)
- [ ] Configure backup for logs
- [ ] Use restart: always
- [ ] Test health checks
- [ ] Review security settings

---

## 📈 Monitoring

### Prometheus Metrics
```
http://localhost:5000/metrics
```

### Application Health
```
http://localhost:5000/health
```

### API Endpoint
```
http://localhost:5000/api/health
```

---

## 🔄 Backup

### Backup logs
```bash
tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/
```

### Backup configuration
```bash
cp .env .env.backup
```

---

## 📚 Documentation

- Full guide: `DOCKER_DEPLOYMENT.md`
- Main README: `README.md`

---

**Need help?** Check the full deployment guide or open an issue!

