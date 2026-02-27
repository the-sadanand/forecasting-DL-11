FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML libraries and Prophet
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 libstdc++6 libblas3 liblapack3 libgfortran5 \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies (pre-built wheels for speed)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy cython && \
    pip install --no-cache-dir \
    torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-initialize cmdstanpy Stan backend (required for Prophet)
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()" || true

# Set environment variables for Stan
ENV STAN_BACKEND=cmdstanpy
ENV CMDSTANPY_USE_INSTALLED=1

# Create directories
RUN mkdir -p data/raw data/processed results logs models && \
    chmod -R 777 data results logs models

# Copy source code and configuration
COPY src/ ./src/
COPY verify_project.py .
COPY entrypoint.sh .

# Copy raw data if available
COPY data/raw/ ./data/raw/

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Run
ENTRYPOINT ["./entrypoint.sh"]