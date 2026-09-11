"""CyberTGN Production Service & SOC Operations API.

Provides:
- Legacy inference routes: /predict, /predict/sequence, /model, /health
- Real-time packet capture: /sniff/start, /sniff/stop, /sniff/status
- Offline forensic analysis: /analyze/pcap
- Automated defense: /alerts, /mitigation/unblock, /model/reset
- SOC Web Dashboard: /dashboard
- Real-time WebSocket event stream: /ws/traffic
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Literal, List, Dict, Any, Optional, Set

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from model_runtime import ModelRuntime, DEFAULT_CHECKPOINT, FEATURE_NAMES, STAGES
from engine.flow_engine import FlowEngine
from engine.hybrid_detector import HybridDetector
from engine.mitigation import MitigationController
from engine.live_sniffer import LiveTrafficSniffer

ROOT = Path(__file__).resolve().parent
DASHBOARD_HTML = ROOT / "static" / "dashboard.html"

logger = logging.getLogger("cybertgn.api")


# --- Schemas ---

class FlowRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    src_ip: str
    dst_ip: str
    timestamp: float = Field(ge=0, le=1e12, description='Unix seconds; ascending within a sequence')
    features: list[float] = Field(min_length=16, max_length=16, description='Feature order: GET /model')

    @field_validator('src_ip', 'dst_ip')
    @classmethod
    def valid_ip(cls, value):
        return str(ip_address(value))


class SequenceRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    flows: list[FlowRequest] = Field(min_length=1, max_length=1000)
    feature_format: Literal['raw', 'signed_log1p'] = 'raw'
    batch_size: int = Field(default=200, ge=1, le=200, description='Training default 200; 1 is experimental and performed poorly')
    threshold: float = Field(default=0.5, ge=0, le=1)


class Prediction(BaseModel):
    index: int
    attack_probability: float
    is_attack: bool
    mitre_stage: int
    stage_name: str
    stage_probabilities: list[float]
    cold_start: bool


class UnblockRequest(BaseModel):
    ip: str

    @field_validator("ip")
    @classmethod
    def valid_ip(cls, value):
        return str(ip_address(value))


class SniffStartRequest(BaseModel):
    interface: Optional[str] = None
    bpf_filter: str = "ip"


# --- Connection Manager for WebSockets ---

class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        payload = json.dumps(message)
        for ws in tuple(self.active_connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_connections.discard(ws)


# --- App Factory ---

def create_app(checkpoint=None):
    ckpt_path = checkpoint or os.environ.get('TGN_CHECKPOINT', DEFAULT_CHECKPOINT)
    ws_manager = WebSocketManager()
    loop: Optional[asyncio.AbstractEventLoop] = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal loop
        loop = asyncio.get_running_loop()
        app.state.model = ModelRuntime(ckpt_path)
        app.state.hybrid_detector = HybridDetector(checkpoint_path=ckpt_path)
        app.state.mitigation = MitigationController()
        app.state.sniffer = LiveTrafficSniffer(
            detector=app.state.hybrid_detector,
            mitigation=app.state.mitigation
        )

        # Connect live sniffer events to WebSocket broadcast
        def on_sniffed_flow(alert):
            if loop and loop.is_running() and ws_manager.active_connections:
                payload = {
                    "type": "ALERT",
                    "data": {
                        "timestamp": alert.timestamp,
                        "src_ip": alert.src_ip,
                        "dst_ip": alert.dst_ip,
                        "src_port": alert.src_port,
                        "dst_port": alert.dst_port,
                        "protocol": alert.protocol,
                        "is_attack": alert.is_attack,
                        "final_score": alert.final_score,
                        "detection_source": alert.detection_source,
                        "threat_type": alert.threat_type,
                        "mitre_stage": alert.mitre_stage,
                        "explanation": alert.explanation,
                        "cold_start": alert.cold_start
                    }
                }
                asyncio.run_coroutine_threadsafe(ws_manager.broadcast(payload), loop)

        app.state.sniffer.on_flow_scored = on_sniffed_flow

        yield

        if app.state.sniffer.is_running:
            app.state.sniffer.stop()

    app = FastAPI(
        title='CyberTGN Complete Threat Defense Platform',
        version='2.0.0',
        lifespan=lifespan,
        description='Temporal Graph Network & Fast-Path Hybrid Threat Detection and Automated Mitigation.'
    )

    # --- Core Baseline Routes ---

    @app.get('/health')
    def health(request: Request):
        return {'status': 'ok', 'checkpoint_sha256': request.app.state.model.sha256}

    @app.get('/model')
    def model(request: Request):
        runtime = request.app.state.model
        return dict(
            checkpoint=runtime.path.name,
            sha256=runtime.sha256,
            recorded_training_metrics=runtime.metadata,
            feature_order=FEATURE_NAMES,
            transform='sign(x) * log1p(abs(x))',
            classes=runtime.num_classes,
            stage_names={i: STAGES[i] if i < len(STAGES) else f'unmapped_{i}' for i in range(runtime.num_classes)},
            max_sequence_length=1000,
            default_batch_size=200,
            state_policy='reset per request; predict then update each batch',
            current_flow_features=runtime.current_flow_head is not None,
            timestamp_encoding=runtime.timestamp_encoding,
            stage_training_counts=runtime.stage_training_counts,
            limitations=[
                ('First batch uses current-flow features but has no historical evidence.'
                 if runtime.current_flow_head is not None else
                 'First batch has no historical evidence; current features affect later batches.'),
                ('Validate the serving batch size and threshold using this candidate’s operating-point report.'
                 if runtime.current_flow_head is not None else
                 'Training default batch size is 200. Batch size 1 performed poorly in the sample evaluation.'),
                ('Stage quality requires independent labeled evaluation; consult stage_training_counts for training support.'
                 if runtime.current_flow_head is not None else
                 'Stage IDs 8-13 have no mapping in the supplied training code.'),
                'Validation data was used for checkpoint selection; independent test required.'
            ]
        )

    def run_sequence(body, request):
        try:
            return request.app.state.model.predict_sequence(
                [f.model_dump() for f in body.flows],
                body.feature_format,
                body.threshold,
                body.batch_size
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post('/predict/sequence', response_model=list[Prediction])
    def sequence(body: SequenceRequest, request: Request):
        return run_sequence(body, request)

    @app.post('/predict', response_model=Prediction,
              description='One raw-feature flow with empty history. Prefer /predict/sequence.')
    def predict(body: FlowRequest, request: Request):
        return run_sequence(SequenceRequest(flows=[body]), request)[0]

    # --- New Operational & Forensic Routes ---

    @app.get('/dashboard', response_class=HTMLResponse)
    def get_dashboard():
        """Serves the interactive SOC Operations Dashboard."""
        if DASHBOARD_HTML.exists():
            return HTMLResponse(content=DASHBOARD_HTML.read_text(encoding='utf-8'))
        return HTMLResponse(content="<h2>Dashboard not found. Ensure static/dashboard.html exists.</h2>", status_code=404)

    @app.post('/analyze/pcap')
    def analyze_pcap(request: Request, file: UploadFile = File(...)):
        """Uploads and analyzes a PCAP file using the Bidirectional FlowEngine + HybridDetector."""
        if not (file.filename or '').lower().endswith(('.pcap', '.pcapng')):
            raise HTTPException(status_code=400, detail="Uploaded file must be .pcap or .pcapng")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
                tmp_path = tmp.name
                size = 0
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 64 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="PCAP upload exceeds 64 MiB")
                    tmp.write(chunk)
            flow_engine = FlowEngine()
            flows = flow_engine.process_pcap_file(tmp_path)

            if not flows:
                return {
                    "filename": file.filename,
                    "flows_analyzed": 0,
                    "threats_detected": 0,
                    "results": [],
                    "message": "No IP flows were found in the uploaded PCAP file."
                }

            # Run dual hybrid analysis
            detector = HybridDetector(checkpoint_path=request.app.state.model.path)
            alerts = detector.analyze_flows(flows)

            results_payload = []
            threats_count = 0
            for alert in alerts:
                if alert.is_attack:
                    threats_count += 1

                results_payload.append({
                    "index": alert.index,
                    "timestamp": alert.timestamp,
                    "src_ip": alert.src_ip,
                    "dst_ip": alert.dst_ip,
                    "src_port": alert.src_port,
                    "dst_port": alert.dst_port,
                    "protocol": alert.protocol,
                    "is_attack": alert.is_attack,
                    "final_score": alert.final_score,
                    "detection_source": alert.detection_source,
                    "threat_type": alert.threat_type,
                    "mitre_stage": alert.mitre_stage,
                    "explanation": alert.explanation,
                    "cold_start": alert.cold_start
                })

            return {
                "filename": file.filename,
                "flows_analyzed": len(flows),
                "threats_detected": threats_count,
                "results": results_payload
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to process PCAP file {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Unable to analyze the uploaded capture")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.post('/sniff/start')
    def start_sniff(request: Request, body: SniffStartRequest = SniffStartRequest()):
        """Starts live background packet sniffing."""
        sniffer: LiveTrafficSniffer = request.app.state.sniffer
        if sniffer.is_running:
            return {"status": "already_running", "info": sniffer.get_status()}
        sniffer.interface = body.interface
        sniffer.bpf_filter = body.bpf_filter
        sniffer.start()
        return {"status": "started", "info": sniffer.get_status()}

    @app.post('/sniff/stop')
    def stop_sniff(request: Request):
        """Stops live background packet sniffing."""
        sniffer: LiveTrafficSniffer = request.app.state.sniffer
        sniffer.stop()
        return {"status": "stopped", "info": sniffer.get_status()}

    @app.get('/sniff/status')
    def sniff_status(request: Request):
        """Returns the status and telemetry of the live packet sniffer."""
        return request.app.state.sniffer.get_status()

    @app.get('/alerts')
    def get_alerts(request: Request, limit: int = Query(default=50, ge=0, le=500)):
        """Retrieves active firewall blocks and recent incident records."""
        mitigation: MitigationController = request.app.state.mitigation
        return {
            "blocked_ips": mitigation.get_active_blocks(),
            "recent_incidents": mitigation.get_recent_incidents(limit=limit)
        }

    @app.post('/mitigation/unblock')
    def unblock_host(request: Request, body: UnblockRequest):
        """Manually unblocks an IP from the firewall blocklist."""
        unblocked = request.app.state.mitigation.unblock_ip(body.ip)
        return {"ip": body.ip, "unblocked": unblocked}

    @app.post('/model/reset')
    def reset_model(request: Request):
        """Resets persistent CyberTGN graph memory, IP mapper, and heuristics."""
        request.app.state.hybrid_detector.reset_state()
        return {"status": "reset_complete"}

    @app.websocket('/ws/traffic')
    async def websocket_traffic(websocket: WebSocket):
        """Streams real-time flow alerts directly to the SOC Dashboard."""
        await ws_manager.connect(websocket)
        try:
            while True:
                # Keepalive loop
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception:
            ws_manager.disconnect(websocket)

    return app


app = create_app()

if __name__ == '__main__':
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
