FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy src.py to root for import compatibility
COPY app/src.py ./src.py

# Copy specific model directories
COPY models/precipitation_fall/ ./models/precipitation_fall/
COPY models/rain_or_not/ ./models/rain_or_not/

# Copy data directory if needed
COPY data/ ./data/

# Debug: Comprehensive structure check
RUN echo "=== Complete Directory Structure ===" && \
    ls -la && \
    echo "" && \
    echo "=== App Directory Contents ===" && \
    ls -la app/ && \
    echo "" && \
    echo "=== Models Directory Contents ===" && \
    ls -la models/ && \
    echo "" && \
    echo "=== Rain Model Directory Contents ===" && \
    ls -la models/rain_or_not/ 2>/dev/null || echo "❌ Rain model directory not found or empty" && \
    echo "" && \
    echo "=== Precipitation Model Directory Contents ===" && \
    ls -la models/precipitation_fall/ 2>/dev/null || echo "❌ Precipitation model directory not found or empty" && \
    echo "" && \
    echo "=== All .pkl files in the container ===" && \
    find . -name "*.pkl" -type f 2>/dev/null || echo "❌ No .pkl files found" && \
    echo "" && \
    echo "=== Python files that can be imported ===" && \
    find . -name "*.py" -type f | grep -E "(src|main)" && \
    echo "=== Structure verification complete ==="

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]