# CyberTGN trained model

This folder contains the inference application and the best checkpoint at
`checkpoints/tgn_best.pt` (epoch index 5). Training scripts, datasets, archives,
and unused demo components have been removed.

## Start the application

From this directory, using the existing environment:

```bash
venv/bin/python app.py
```

Open http://127.0.0.1:8000/docs for the interactive request editor. Expand
`POST /predict/sequence`, select **Try it out**, paste `examples/flows.json`,
and execute. `GET /model` gives the feature order and model limitations.

For a new environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-inference.txt
.venv/bin/python app.py
```

The application also supports local JSON prediction:

```bash
venv/bin/python app.py --input examples/flows.json --output predictions.json
```

## Use the API from another program

The separate API entry point runs the same checkpoint-backed model:

```bash
venv/bin/python api.py --host 0.0.0.0 --port 8000
```

Other computers that can reach this host can use its IP address on port 8000.
This command starts a local/LAN service; no public internet deployment was made.
The service has no built-in authentication. For public hosting, use HTTPS,
authentication, request-size limits and rate limits at your hosting gateway.

```bash
curl http://127.0.0.1:8000/predict/sequence \
  -H 'Content-Type: application/json' \
  --data-binary @examples/flows.json
```

Python client using only the standard library:

```python
import json
from pathlib import Path
from urllib.request import Request, urlopen

request = Request(
    'http://127.0.0.1:8000/predict/sequence',
    data=Path('examples/flows.json').read_bytes(),
    headers={'Content-Type': 'application/json'},
)
with urlopen(request, timeout=60) as response:
    predictions = json.load(response)
print(predictions)
```

Endpoints: `/health`, `/model`, `/predict`, `/predict/sequence`, `/docs`,
`/openapi.json`. Both `uvicorn app:app` and `uvicorn api:app` are supported.
Set `TGN_CHECKPOINT` for the `api.py`/uvicorn entry point or use
`app.py --checkpoint PATH` to select another trusted checkpoint. Model loading
fails at startup if the file is missing or incompatible; there is no random
weight fallback. Runtime uses CPU and loads weights with `weights_only=True`.

## Input and temporal behavior

A sequence contains 1–1000 flows, ascending Unix timestamps in seconds, source
and destination IP addresses, and exactly 16 finite numbers per flow. The
`feature_format` field is `raw` by default. Use `signed_log1p` only for features
that are already transformed with `sign(x) * log1p(abs(x))`.

Feature order (original CICFlowMeter units, including microsecond duration/IAT
where supplied by that exporter):

1. Flow Duration
2. Total Fwd Packet
3. Total Bwd packets
4. Total Length of Fwd Packet
5. Total Length of Bwd Packet
6. Fwd Packet Length Mean
7. Bwd Packet Length Mean
8. Flow Bytes/s
9. Flow Packets/s
10. Flow IAT Mean
11. Fwd IAT Mean
12. Bwd IAT Mean
13. Fwd Packets/s
14. Bwd Packets/s
15. Average Packet Size
16. Down/Up Ratio

Each request starts with empty graph memory and a fresh collision-free IP
mapping. Flows are scored in batches before updating memory with their features.
`batch_size` defaults to 200 to match the archived training default. Use more
than 200 flows to give later batches historical context. `batch_size: 1` is an
experimental mode that performed poorly on the sampled validation data. Calls are serialized around model state, so callers
do not contaminate one another. Include relevant history in each request;
there is no persistent session across requests or server restarts.

The **first batch has no historical evidence** and its predictions do not depend
on the current feature vectors. This follows the trained architecture. `/predict`
accepts one raw-feature flow for compatibility, but its result alone is not a
meaningful test of traffic detection. `/predict/sequence` is the primary route.
Batch size affects predictions: the original training default was 200, while
the API now defaults to the same size; an explicit batch size of 1 updates
after every flow. Partial final batches are supported.

`attack_probability` is the sigmoid model score, not a calibrated probability
of real-world harm. `is_attack` uses threshold 0.5 unless overridden. Stage
classification is a separate head and can disagree with attack detection.
Stage IDs 0–7 follow the archived mapper; IDs 8–13 are returned as `unmapped_N`.
The measured stage classifier is unreliable; see the evaluation report.

Synthetic example predictions only demonstrate operation, not accuracy.

## Verify the service

```bash
venv/bin/python -m unittest discover -s tests -p test_model_service.py -v
```

## Saved evaluation results

The `reports/` folder retains the historical scores, per-chunk results,
checkpoint hashes, and evaluation limitations. Paths in the JSON reports refer
to the original training layout. The best checkpoint has been moved without
changing its contents. Training data and the evaluator have been removed, so
these dataset evaluations cannot be rerun from this folder alone.

The best model measured attack precision 76.35%, recall 96.20%, and F1 0.8513
on long validation chunks. Short sequences performed poorly, and attack-stage
classification was unreliable. See `reports/MODEL_REPORT.md` for details.

## Share the application

Copy `api.py`, `app.py`, `model_runtime.py`, `requirements-inference.txt`,
`models/`, and `checkpoints/`. Include `examples/` and this guide for usage.
Install the requirements in a new Python 3.12 environment on the destination;
the local `venv/` is retained for running on this machine.
