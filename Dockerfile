FROM python:3.12-slim

LABEL maintainer="Champa Intelligence <info@champa-bpmn.com>"
LABEL description="Camunda 7 Health Monitor - Lightweight monitoring dashboard"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY config.py .
COPY wsgi.py .
COPY gunicorn.conf.py .
COPY templates/ templates/
COPY helpers/ helpers/
COPY routes/ routes/
COPY services/ services/

# Create logs directory with proper permissions
RUN mkdir -p logs

# Create non-root user for security
RUN useradd -m -u 1000 camunda && \
    chown -R camunda:camunda /app

USER camunda

# Expose port
EXPOSE 5000

# Health check using Kubernetes liveness probe endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health/live || exit 1

# Default environment variables (can be overridden)
ENV PORT=5000 \
    DEBUG=false \
    JSON_LOGGING=false \
    SSL_VERIFY=false

# Run with Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]