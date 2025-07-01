FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create non-root user
RUN useradd -m -s /bin/bash memorygate
USER memorygate
WORKDIR /home/memorygate

# Initialize Python environment
ARG PYTHONPATH_ARG=.
ENV PYTHONPATH=${PYTHONPATH_ARG}:${PYTHONPATH}
ENV PYTHONUNBUFFERED=1

# Install Python dependencies - actual installation happens in postCreateCommand
COPY pyproject.toml ./

# Create cache directories for pytest and ruff
RUN mkdir -p /home/memorygate/.cache/pytest \
    && mkdir -p /home/memorygate/.cache/ruff \
    && mkdir -p /home/memorygate/.cache/torch \
    && mkdir -p /home/memorygate/.cache/huggingface
