FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY gunicorn.conf.py .
COPY templates/ templates/

# Create logs directory
RUN mkdir -p logs

# Create non-root user
RUN useradd -m -u 1000 camunda && chown -R camunda:camunda /app
USER camunda

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health', timeout=5)"

# Run with Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]