"""CyberTGN application: serve an interactive API or predict from a JSON file."""
import argparse
import json
from pathlib import Path
from api import app, SequenceRequest
from model_runtime import DEFAULT_CHECKPOINT, ModelRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='SequenceRequest JSON file; omit to open the API server')
    parser.add_argument('--output', type=Path, help='Write predictions as JSON')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.input:
        body = SequenceRequest.model_validate_json(args.input.read_text())
        predictions = ModelRuntime(args.checkpoint).predict_sequence(
            [f.model_dump() for f in body.flows], body.feature_format, body.threshold, body.batch_size)
        result = json.dumps(predictions, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(result + '\n')
        else:
            print(result)
    else:
        import uvicorn
        from api import create_app
        uvicorn.run(create_app(args.checkpoint), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
