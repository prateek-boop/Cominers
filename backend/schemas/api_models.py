from pydantic import BaseModel
from typing import List, Optional

class IngestResponse(BaseModel):
    status: str
    events_processed: int

class PredictionResponse(BaseModel):
    source_ip: str
    dest_ip: str
    attack_probability_15s: float
    attack_probability_30s: float
    attack_probability_45s: float
    mitre_stage: str
    conformal_set: List[str]

class AlertResponse(BaseModel):
    alert_id: str
    prediction: PredictionResponse
    action_taken: str

class ExplainResponse(BaseModel):
    alert_id: str
    top_features: dict
    important_nodes: List[str]
