from fastapi import APIRouter, BackgroundTasks
from typing import List
from backend.schemas.api_models import IngestResponse, PredictionResponse, AlertResponse, ExplainResponse

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_data(background_tasks: BackgroundTasks):
    return IngestResponse(status="started", events_processed=0)

@router.get("/prediction", response_model=List[PredictionResponse])
async def get_predictions():
    return []

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts():
    return []

@router.get("/explain/{alert_id}", response_model=ExplainResponse)
async def explain_alert(alert_id: str):
    return ExplainResponse(alert_id=alert_id, top_features={}, important_nodes=[])
    
@router.post("/response/block")
async def block_ip(ip: str):
    return {"status": "blocked", "ip": ip}
