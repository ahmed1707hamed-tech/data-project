# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install necessary system packages
# We clean up the apt cache to keep the image size small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Using tensorflow-cpu in requirements to save ~1GB of disk space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project context (excluding files in .dockerignore if any)
COPY . .

# Expose the API port
EXPOSE 8000

# Create a non-root user for security (Best Practice)
RUN adduser --disabled-password --gecos '' appuser
# Give ownership of the app directory to the new user
RUN chown -R appuser:appuser /app
USER appuser

# Run FastAPI using Gunicorn as the process manager with Uvicorn workers
# We use 4 workers as a starting point. This can be tuned based on Lightning AI's CPU count.
CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
