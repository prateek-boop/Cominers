# Cyber Attack Forecasting System

A Temporal Graph Neural Network (TGN) system for real-time cyber attack prediction, MITRE ATT&CK classification, and automated response.

## Architecture

This project processes raw PCAP or NetFlow data into a continuous-time dynamic graph, then uses a Temporal Graph Network (TGN) via PyTorch Geometric to predict future network states and classify attack stages based on the MITRE ATT&CK framework.

## Requirements
* Docker and Docker Compose
* NVIDIA Container Toolkit (for GPU-accelerated training)

## How to Run

### Development & API (Local)
To start the database and the FastAPI backend without running the heavy training loop:
```bash
docker-compose up api db
```

### Training (External System / GPU Server)
This project is containerized for seamless training on external GPU systems. Transfer this repository to the target machine and run:
```bash
# This uses the "training" profile to run the GPU-enabled trainer service
docker-compose --profile training up trainer
```

### Structure
* `ingestion/`: PCAP/CSV parsers and data normalizers
* `graph/`: Temporal network graph builder
* `models/`: PyTorch Geometric TGN implementation
* `forecasting/`: GRU-based delta predictor
* `scripts/`: Training and evaluation scripts
