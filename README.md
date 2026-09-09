# CyberTGN model application

Run the retained best checkpoint through a local CLI or FastAPI service.

```bash
venv/bin/python app.py
# Interactive API: http://127.0.0.1:8000/docs

venv/bin/python app.py --input examples/flows.json --output predictions.json
```

For a fresh installation, use Python 3.12 and install `requirements-inference.txt`
in a virtual environment. See [MODEL_USAGE.md](MODEL_USAGE.md) for input format,
API usage, temporal behavior, and known model limitations.

- `checkpoints/tgn_best.pt`: retained best model, unchanged.
- `models/`: the three model components required for inference.
- `api.py`, `app.py`, `model_runtime.py`: application and inference runtime.
- `examples/`: sample request.
- `tests/`: service checks using the real checkpoint.
- `reports/`: historical evaluation scores and limitations.
- `venv/`: existing local Python environment.

Training scripts, datasets, archives, and unused demo components have been
removed. The saved dataset evaluation scores cannot be reproduced without
restoring the evaluation code and data.
