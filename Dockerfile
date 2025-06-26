# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create and set the working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
# Install project dependencies, including optional 'dev', 'gpu', and 'storage'
RUN uv pip install --system -e .[dev,gpu,storage]

# Copy the rest of the application code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m -u 1000 appuser
USER appuser

# Expose the port the app runs on (if any, for now a common one)
EXPOSE 8000

# Default command to run when starting the container (can be overridden)
# For now, it's just keeping the container running.
# A real application entrypoint (e.g., src/memory_gate/main.py) will be added later.
CMD ["tail", "-f", "/dev/null"]
