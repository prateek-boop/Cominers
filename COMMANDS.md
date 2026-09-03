# Cyber Attack Forecasting — Command Reference

This document contains all the essential commands for setting up, training, testing, and running the Temporal Graph Network (TGN) system.

## 🐳 Docker & Orchestration

### 1. Run the Training Pipeline (External GPU System)
This isolates the PyTorch training loop and grants it access to NVIDIA GPUs via the `training` profile:
```bash
docker-compose --profile training up trainer
```

### 2. Run the API and Database (Local / Inference)
Starts PostgreSQL and the FastAPI backend without running the heavy training loop:
```bash
docker-compose up api db
```
*(Add `-d` to the end of the command to run in detached/background mode).*

### 3. Stop All Services
```bash
docker-compose down
```

### 4. Wipe Database Volumes (Hard Reset)
```bash
docker-compose down -v
```

---

## 🐍 Backend & Python Scripts (Local Development)

### 1. Install Dependencies Locally (Optional)
If you want to run scripts outside of Docker:
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
pytest tests/ -v --cov=.
```

### 3. Run FastAPI Server Manually
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🖥️ Frontend (React SOC Dashboard)

### 1. Install Node Modules
```bash
cd frontend/soc-dashboard
npm install
```

### 2. Start Development Server
Runs the Vite dev server with hot module replacement (HMR):
```bash
cd frontend/soc-dashboard
npm run dev
```

### 3. Build for Production
```bash
cd frontend/soc-dashboard
npm run build
```

---

## 📊 Dataset Acquisition

### Download CIC-IDS-2018 (AWS Open Data)
The fastest way to download the raw dataset (requires AWS CLI):
```bash
aws s3 sync s3://cse-cic-ids2018/ data/raw/ --no-sign-request
```

---

## 🛡️ SOAR / Firewall Testing

The `FirewallController` (Phase 10) uses `nftables` to isolate threats. It defaults to `dry_run=True` so it logs commands without executing them on your host network. 

To test it manually in a Python shell:
```python
from response.firewall_controller import FirewallController

fw = FirewallController(dry_run=True)
fw.block_ip("192.168.1.50")
fw.isolate_host()
```
