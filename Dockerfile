# MOEX Agent v2 Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY moex_agent/ ./moex_agent/
COPY db/ ./db/
COPY config.yaml .

# Create data directories
RUN mkdir -p data models

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
ENTRYPOINT ["python", "-m", "moex_agent"]
CMD ["status"]
