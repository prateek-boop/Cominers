"""Run with: venv/bin/python api.py; interactive client at /docs."""
from contextlib import asynccontextmanager
from ipaddress import ip_address
import os
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from model_runtime import ModelRuntime, DEFAULT_CHECKPOINT, FEATURE_NAMES, STAGES


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


def create_app(checkpoint=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model = ModelRuntime(checkpoint or os.environ.get('TGN_CHECKPOINT', DEFAULT_CHECKPOINT))
        yield
        del app.state.model

    app = FastAPI(title='CyberTGN Model API', version='1.0.0', lifespan=lifespan,
        description='Checkpoint-backed attack scoring. Each request starts a new graph. '
        'Use /predict/sequence for history-aware predictions. Scores are not calibrated risk estimates.')

    @app.get('/health')
    def health(request: Request):
        return {'status': 'ok', 'checkpoint_sha256': request.app.state.model.sha256}

    @app.get('/model')
    def model(request: Request):
        runtime = request.app.state.model
        return dict(checkpoint=runtime.path.name, sha256=runtime.sha256,
            recorded_training_metrics=runtime.metadata, feature_order=FEATURE_NAMES,
            transform='sign(x) * log1p(abs(x))', classes=runtime.num_classes,
            stage_names={i: STAGES[i] if i < len(STAGES) else f'unmapped_{i}' for i in range(runtime.num_classes)},
            max_sequence_length=1000, default_batch_size=200, state_policy='reset per request; predict then update each batch',
            limitations=['First batch has no historical evidence; current features affect later batches.',
                        'Training default batch size is 200. Batch size 1 performed poorly in the sample evaluation.',
                        'Stage IDs 8-13 have no mapping in the supplied training code.',
                        'Validation data was used for checkpoint selection; independent test required.'])

    def run(body, request):
        try:
            return request.app.state.model.predict_sequence(
                [f.model_dump() for f in body.flows], body.feature_format, body.threshold, body.batch_size)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post('/predict/sequence', response_model=list[Prediction])
    def sequence(body: SequenceRequest, request: Request):
        return run(body, request)

    @app.post('/predict', response_model=Prediction,
              description='One raw-feature flow with empty history. Prefer /predict/sequence.')
    def predict(body: FlowRequest, request: Request):
        return run(SequenceRequest(flows=[body]), request)[0]

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
