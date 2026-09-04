# Use an official PyTorch runtime with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies (libpcap for Scapy)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpcap-dev \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch Geometric dependencies separately 
# (These must match the base image's PyTorch/CUDA version)
RUN pip install --no-cache-dir \
    torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# Copy project configuration
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Add /app to PYTHONPATH
ENV PYTHONPATH=/app

# Default command (can be overridden by docker-compose)
CMD ["python", "scripts/train_tgn.py"]
