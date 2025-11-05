# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for your packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy the ChromaDB database (read-only)
COPY recipes_db/ ./recipes_db/

# Expose port (Cloud Run will override this with PORT env var)
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
